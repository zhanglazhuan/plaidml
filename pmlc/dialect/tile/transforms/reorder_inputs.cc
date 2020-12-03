// Copyright 2020, Intel Corporation

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/SetVector.h"

#include "pmlc/dialect/layer/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/pass_detail.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::tile {

namespace {

SmallVector<Operation *, 4> toRemove;

SmallVector<int64_t, 4> getNewSizes(RankedTensorType &orgTensor,
                                    bool weights = false) {
  SmallVector<int64_t, 4> newSizes;
  if (orgTensor.getRank() == 4) {
    if (weights) {
      newSizes.push_back(orgTensor.getDimSize(2));
      newSizes.push_back(orgTensor.getDimSize(3));
      newSizes.push_back(orgTensor.getDimSize(1));
      newSizes.push_back(orgTensor.getDimSize(0));
    } else {
      newSizes.push_back(orgTensor.getDimSize(0));
      newSizes.push_back(orgTensor.getDimSize(2));
      newSizes.push_back(orgTensor.getDimSize(3));
      newSizes.push_back(orgTensor.getDimSize(1));
    }
  } else {
    newSizes.push_back(orgTensor.getDimSize(0));
    newSizes.push_back(orgTensor.getDimSize(1));
  }
  return newSizes;
}

int64_t getArgPos(Operation &op, BlockArgument &arg) {
  auto argPos = 0;
  for (auto i = 0; i < op.getOperands().size(); i++) {
    if (op.getOperand(i) == arg) {
      argPos = i;
    }
  }
  return argPos;
}

int64_t getOperandArgNum(Operation &op, Value &operand) {
  for (auto i = 0; i < op.getOperands().size(); i++) {
    if (op.getOperand(i) == operand) {
      return i;
    }
  }
  return 0;
}

void performDataReordering(OpBuilder &builder, Operation *op, BlockArgument arg,
                           bool isConst) {
  auto layerOp = dyn_cast<layer::BoxOp>(op);
  if (!layerOp)
    return;

  auto argPos = getArgPos(*op, arg);
  auto tensorType =
      op->getOperand(argPos).getType().dyn_cast<mlir::RankedTensorType>();
  if (!tensorType)
    return;

  Type elementType = tensorType.getElementType();
  auto ident = tile::createIdentity(builder, op->getLoc(), elementType,
                                    AggregationKind::assign);
  auto outRank = tensorType.getRank();
  if (outRank != 4)
    return;

  // TODO: Do proper reordering here, generalize, right now only NCHW->NHWC is
  // supported
  SmallVector<AffineExpr, 4> dimExprs;
  dimExprs.reserve(outRank);
  if (isConst && layerOp.op().compare("ng.Convolution")) {
    dimExprs.push_back(mlir::getAffineDimExpr(2, op->getContext()));
    dimExprs.push_back(mlir::getAffineDimExpr(3, op->getContext()));
    dimExprs.push_back(mlir::getAffineDimExpr(1, op->getContext()));
    dimExprs.push_back(mlir::getAffineDimExpr(0, op->getContext()));
  } else {
    dimExprs.push_back(mlir::getAffineDimExpr(0, op->getContext()));
    dimExprs.push_back(mlir::getAffineDimExpr(2, op->getContext()));
    dimExprs.push_back(mlir::getAffineDimExpr(3, op->getContext()));
    dimExprs.push_back(mlir::getAffineDimExpr(1, op->getContext()));
  }
  auto idMap = AffineMap::get(outRank, 0, dimExprs, op->getContext());

  std::stringstream ss;
  ss << "reordered_" << arg.getArgNumber();

  auto newName = layerOp.op();
  auto newOp = builder.create<ContractionOp>(
      op->getLoc(), op->getOperand(argPos).getType(), ident,
      ArrayRef<Value>{op->getOperand(argPos)}, AggregationKind::assign,
      CombinationKind::none, idMap, ArrayRef<AffineMap>{idMap},
      IntegerSet::getEmptySet(outRank, 0, op->getContext()), ss.str());
  newOp.setLowerBounds(SmallVector<int64_t, 4>(outRank, 0));

  auto isWeight = (isConst && (layerOp.op().compare("ng.Convolution") == 0));
  SmallVector<int64_t, 4> newSizes = getNewSizes(tensorType, isWeight);
  SmallVector<int64_t, 4> newOutSize;
  SmallVector<int64_t, 4> reorderedUpperBounds;
  for (unsigned i = 0; i < outRank; i++) {
    newOutSize.push_back(newSizes[i]);
    reorderedUpperBounds.push_back(newSizes[i] - 1);
  }
  newOp.setUpperBounds(reorderedUpperBounds);

  // Switch all uses to the new contracton
  op->getOperand(argPos).replaceAllUsesExcept(
      newOp.getResult(), SmallPtrSet<Operation *, 1>{newOp.getOperation()});
  // Update output size
  newOp.getResult().setType(RankedTensorType::get(newOutSize, elementType));
}

void performLayersReshape(OpBuilder &builder, Operation *op) {
  auto layerOp = dyn_cast<layer::BoxOp>(op);
  if (!layerOp)
    return;

  // Switch output order of orginal Op
  auto tensorTypeLayer =
      op->getResult(0).getType().dyn_cast<mlir::RankedTensorType>();
  if (!tensorTypeLayer)
    return;

  Type elementTypeLayer = tensorTypeLayer.getElementType();
  auto outRankLayer = tensorTypeLayer.getRank();
  if (outRankLayer < 2)
    return;

  // TODO: Do proper reordering here, generalize, right now only NCHW->NHWC is
  // supported
  SmallVector<int64_t, 4> newSizesOutput = getNewSizes(tensorTypeLayer);

  SmallVector<Value, 8> operands;
  for (auto &operand : layerOp.getOperands())
    operands.push_back(operand);
  SmallVector<Type, 8> resultTypes;
  for (auto &result : layerOp.getResults())
    resultTypes.push_back(result.getType());
  auto newLayerOp = builder.create<layer::BoxOp>(
      layerOp.getLoc(), layerOp.op(), operands, resultTypes,
      builder.getDictionaryAttr(layerOp.getAttrs()));
  newLayerOp.getResult(0).setType(
      RankedTensorType::get(newSizesOutput, elementTypeLayer));

  // Move inner body of old op to the new one
  auto orgBody = layerOp.getBody();
  auto &newLayerOps = newLayerOp.getBody()->getOperations();
  auto &layerOps = layerOp.getBody()->getOperations();
  newLayerOps.splice(std::prev(newLayerOps.end()), layerOps,
                     std::next(layerOps.begin(), 0), layerOps.end());

  op->getResult(0).replaceAllUsesWith(newLayerOp.getResult(0));

  auto origNumArgs = layerOp.getBody()->getArguments().size();
  auto curArgNum = 0;
  for (size_t i = 0; i < origNumArgs; i++) {
    auto curArg = layerOp.getBody()->getArgument(curArgNum);
    curArg.replaceAllUsesWith(newLayerOp.getBody()->getArgument(curArgNum));
    curArgNum++;
  }

  auto layerReturnOp = dyn_cast<layer::ReturnOp>(newLayerOps.back());
  if (!layerReturnOp)
    return;
  layerReturnOp.getOperand(0).setType(
      RankedTensorType::get(newSizesOutput, elementTypeLayer));

  for (auto &op : newLayerOps) {
    for (auto &result : op.getResults()) {
      auto tensorTypeOperand =
          result.getType().dyn_cast<mlir::RankedTensorType>();
      if (!tensorTypeOperand)
        continue;
      Type elementTypeOperand = tensorTypeOperand.getElementType();

      if (tensorTypeOperand.getRank() < 2)
        continue;

      auto newSizes = getNewSizes(tensorTypeLayer);
      result.setType(RankedTensorType::get(newSizes, elementTypeOperand));

      auto contractionOp = dyn_cast<ContractionOp>(op);
      if (contractionOp) {
        if (layerOp.op().compare("ng.Convolution") == 0) {
          auto dict = layerOp.getAttrOfType<DictionaryAttr>("attrs");
          if (!dict)
            return;

          auto padsBegin = dict.get("pads_begin").dyn_cast_or_null<ArrayAttr>();
          SmallVector<int64_t, 4> padsBeginVec;
          for (auto pad : padsBegin) {
            if (auto integerAttr = pad.dyn_cast<IntegerAttr>())
              padsBeginVec.push_back(integerAttr.getInt());
          }

          auto strides = dict.get("strides").dyn_cast_or_null<ArrayAttr>();
          SmallVector<int64_t, 4> stridesVec;
          for (auto stride : strides) {
            if (auto integerAttr = stride.dyn_cast<IntegerAttr>())
              stridesVec.push_back(integerAttr.getInt());
          }

          SmallVector<AffineExpr, 4> dimExprs;
          dimExprs.push_back(mlir::getAffineDimExpr(0, op.getContext()));
          dimExprs.push_back(
              mlir::getAffineDimExpr(1, op.getContext()) * stridesVec[0] +
              mlir::getAffineDimExpr(4, op.getContext()) - padsBeginVec[0]);
          dimExprs.push_back(
              mlir::getAffineDimExpr(2, op.getContext()) * stridesVec[1] +
              mlir::getAffineDimExpr(5, op.getContext()) - padsBeginVec[1]);
          dimExprs.push_back(mlir::getAffineDimExpr(6, op.getContext()));

          auto newMap0 = AffineMap::get(7, 0, dimExprs, op.getContext());

          SmallVector<AffineExpr, 4> dimExprs1;
          dimExprs1.push_back(mlir::getAffineDimExpr(4, op.getContext()));
          dimExprs1.push_back(mlir::getAffineDimExpr(5, op.getContext()));
          dimExprs1.push_back(mlir::getAffineDimExpr(6, op.getContext()));
          dimExprs1.push_back(mlir::getAffineDimExpr(3, op.getContext()));

          auto newMap1 = AffineMap::get(7, 0, dimExprs1, op.getContext());

          contractionOp.setSources(ArrayRef<AffineMap>{newMap0, newMap1});
        } else if (layerOp.op().compare("ng.MaxPool") == 0) {
          auto dict = layerOp.getAttrOfType<DictionaryAttr>("attrs");
          if (!dict)
            return;

          auto padsBegin = dict.get("pads_begin").dyn_cast_or_null<ArrayAttr>();
          SmallVector<int64_t, 4> padsBeginVec;
          for (auto pad : padsBegin) {
            if (auto integerAttr = pad.dyn_cast<IntegerAttr>())
              padsBeginVec.push_back(integerAttr.getInt());
          }

          auto strides = dict.get("strides").dyn_cast_or_null<ArrayAttr>();
          SmallVector<int64_t, 4> stridesVec;
          for (auto stride : strides) {
            if (auto integerAttr = stride.dyn_cast<IntegerAttr>())
              stridesVec.push_back(integerAttr.getInt());
          }

          SmallVector<AffineExpr, 4> dimExprs;
          dimExprs.push_back(mlir::getAffineDimExpr(0, op.getContext()));
          dimExprs.push_back(
              mlir::getAffineDimExpr(1, op.getContext()) * stridesVec[0] +
              mlir::getAffineDimExpr(4, op.getContext()) - padsBeginVec[0]);
          dimExprs.push_back(
              mlir::getAffineDimExpr(2, op.getContext()) * stridesVec[1] +
              mlir::getAffineDimExpr(5, op.getContext()) - padsBeginVec[1]);
          dimExprs.push_back(mlir::getAffineDimExpr(3, op.getContext()));

          auto newMap0 = AffineMap::get(6, 0, dimExprs, op.getContext());

          contractionOp.setSources(ArrayRef<AffineMap>{newMap0});
        }
      }
    }
  }

  toRemove.push_back(op);
}

void performReordering(layer::BoxOp &LayerOp, FuncOp &func) {
  auto size = func.getArguments().size();

  OpBuilder builder(LayerOp.getOperation());

  for (auto &arg : func.getArguments()) {
    for (auto *op : arg.getUsers()) {
      performDataReordering(
          builder, op, arg,
          static_cast<bool>(func.getArgAttr(arg.getArgNumber(), "tile.const")));
    }
  }

  for (Operation &op : make_early_inc_range(func.getOps())) {
    performLayersReshape(builder, &op);
  }

  for (auto op : make_early_inc_range(toRemove)) {
    op->erase();
  }
}

struct ReorderInputsPass : public ReorderInputsBase<ReorderInputsPass> {
  void runOnFunction() final {
    auto func = getFunction();

    // TODO: this is bad design
    auto &block = func.getBody().front();
    Operation *op = &block.front();
    auto layerOp = dyn_cast<layer::BoxOp>(op);
    if (!layerOp)
      return;

    auto dict = layerOp.getAttrOfType<DictionaryAttr>("attrs");
    if (!dict)
      return;

    auto reorderAttr = dict.get("reorder_input");
    if (!reorderAttr)
      return;

    performReordering(layerOp, func);
  }
};

} // namespace

std::unique_ptr<Pass> createReorderInputsPass() {
  return std::make_unique<ReorderInputsPass>();
}

} // namespace pmlc::dialect::tile
