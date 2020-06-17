#include "plaidml/op/lib/fft.h"

#include <algorithm>
#include <set>
#include <utility>
#include <vector>

#include "llvm/Support/FormatVariadic.h"

#include "plaidml/op/op.h"
#include "pmlc/util/logging.h"

using namespace plaidml::edsl;  // NOLINT
using namespace plaidml::op;    // NOLINT

namespace plaidml::op::lib {

template <typename T>

Buffer makeBuffer(const TensorShape& shape, const std::vector<T>& data) {
  const auto& curDevice = plaidml::Settings::get("PLAIDML_DEVICE");
  Buffer buffer(curDevice, shape);
  buffer.copy_from(data.data());
  return buffer;
}

// fft2_start
int BitReverse(int log2n, int value) {
  int reversed = 0;
  for (int j = 0; j < log2n; j++) {
    reversed <<= 1;
    reversed += value & 1;
    value >>= 1;
  }
  return reversed;
}

// Initialize a full complex twiddle matrix of size n
Tensor Twiddle(int log2n) {
  int n = 1 << log2n;
  std::vector<float> tfact(n * 2);
  for (int i = 0; i < n; i++) {
    tfact[i * 2] = cos(-6.283185307179586 / n);
    tfact[i * 2 + 1] = sin(-6.283185307179586 / n);
  }
  auto bufferA = makeBuffer(TensorShape(DType::INT32, {n, 2}), tfact);
  return Constant(LogicalShape(DType::INT32, {n, 2}), bufferA, "twid");
}

Tensor BitReverse(int log2n, const Tensor& c_in) {
  TensorDim X, CMP;
  TensorIndex x, cmp;
  c_in.bind_dims(X, CMP);
  auto O = TensorOutput(X, CMP);
  int n = 1 << log2n;
  for (int i = 0; i < n; i++) {
    int brn = BitReverse(log2n, i);
    O(x * n + brn, cmp) += c_in(x * n + i, cmp);
  }
  return O;
}

Tensor RFFT2(int log2n, int cur, const Tensor& twiddle, const Tensor& rev) {
  if (cur > log2n) return rev;
  TensorDim X, CMP;
  TensorIndex x, cmp, ridx;
  rev.bind_dims(X, CMP);
  auto O = TensorOutput(X, CMP);
  auto O1 = TensorOutput(X);
  auto O2 = TensorOutput(X);

  int round_sz = 1 << cur;
  int twid_inc = (1 << log2n) / round_sz;

  O1(x) += twiddle(ridx * twid_inc, cmp) * rev(x * round_sz + ridx, cmp);
  O1.add_constraint(ridx < round_sz);
  O1 = -O1;
  O2(x) += twiddle(ridx * twid_inc, cmp) * rev(x * round_sz + ridx, -(cmp - 1));
  O2.add_constraint(ridx < round_sz);
  O(x, 0) += O1(x);
  O(x, 1) += O2(x);
  return RFFT2(log2n, cur + 1, twiddle, O);
}

Tensor FFT2(int log2n, const Tensor& c_in) {
  Tensor twid = Twiddle(log2n);
  Tensor rev = BitReverse(log2n, c_in);
  return RFFT2(log2n, 1, twid, rev);
}

Value fft(const Value& value) {
  IVLOG(1, "reorg_yolo");

  auto args = value.as_tuple();
  if (args.size() != 3) {
    throw std::runtime_error(llvm::formatv("PlaidML fft op expects 3 arguments (received {0})", args.size()));
  }
  auto I = args[0].as_tensor();
  auto radix = args[1].as_int();
  int log2n = std::log2(args[2].as_int());  // check for pow 2 size? rely on padding?
  auto ndims = I.rank();
  if (ndims != 2) {
    throw std::runtime_error(llvm::formatv(
        "PlaidML fft op expects I to be a 2D tensor representing an array of complex numbers (received ndims: {0})",
        ndims));
  } else if (radix != 2) {
    throw std::runtime_error("PlaidML fft op currently only supports radix 2");
  }
  return Value{FFT2(log2n, I)};
}

}  // namespace plaidml::op::lib
