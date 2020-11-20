// Copyright 2020, Intel Corporation

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/pass_detail.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::tile {

namespace {

void performReordering(LayerOp &op) {
  OpBuilder builder(op.getOperation());

  auto tensorType =
      op.getOperand(0).getType().dyn_cast<mlir::RankedTensorType>();
  if (!tensorType)
    return;

  Type elementType = tensorType.getElementType();
  auto ident = tile::createIdentity(builder, op.getLoc(), elementType,
                                    AggregationKind::assign);
  auto outRank = tensorType.getRank();
  if (outRank != 4)
    return;

  // TODO: Do proper reordering here, generalize, right now only NCHW->NHWC is
  // supported
  SmallVector<AffineExpr, 4> dimExprs;
  dimExprs.reserve(outRank);
  dimExprs.push_back(mlir::getAffineDimExpr(0, op.getContext()));
  dimExprs.push_back(mlir::getAffineDimExpr(2, op.getContext()));
  dimExprs.push_back(mlir::getAffineDimExpr(3, op.getContext()));
  dimExprs.push_back(mlir::getAffineDimExpr(1, op.getContext()));
  auto idMap = AffineMap::get(outRank, 0, dimExprs, op.getContext());

  auto newOp = builder.create<ContractionOp>(
      op.getLoc(), op.getOperand(0).getType(), ident,
      ArrayRef<Value>{op.getOperand(0)}, AggregationKind::assign,
      CombinationKind::none, idMap, ArrayRef<AffineMap>{idMap},
      IntegerSet::getEmptySet(outRank, 0, op.getContext()), "reordered_input");
  newOp.setLowerBounds(SmallVector<int64_t, 4>(outRank, 0));

  SmallVector<int64_t, 4> newSizes;
  newSizes[0] = tensorType.getDimSize(0);
  newSizes[1] = tensorType.getDimSize(2);
  newSizes[2] = tensorType.getDimSize(3);
  newSizes[3] = tensorType.getDimSize(1);

  SmallVector<int64_t, 4> newOutSize;
  SmallVector<int64_t, 4> reorderedUpperBounds;
  for (unsigned i = 0; i < outRank; i++) {
    newOutSize.push_back(newSizes[i]);
    reorderedUpperBounds.push_back(newSizes[i] - 1);
  }
  newOp.setUpperBounds(reorderedUpperBounds);

  // Switch all uses to the new contracton
  op.getOperand(0).replaceAllUsesExcept(
      newOp.getResult(), SmallPtrSet<Operation *, 1>{newOp.getOperation()});
  // Update output size
  newOp.getResult().setType(RankedTensorType::get(newOutSize, elementType));
}

struct ReorderInputsPass : public ReorderInputsBase<ReorderInputsPass> {
  void runOnFunction() final {
    auto func = getFunction();
    func.walk([&](LayerOp op) {
      auto dict = op.getAttrOfType<DictionaryAttr>("attrs");
      if (!dict)
        return;

      auto reorderAttr = dict.get("reorder_input");
      if (!reorderAttr)
        return;

      performReordering(op);
    });
  }
};

} // namespace

std::unique_ptr<Pass> createReorderInputsPass() {
  return std::make_unique<ReorderInputsPass>();
}

} // namespace pmlc::dialect::tile
