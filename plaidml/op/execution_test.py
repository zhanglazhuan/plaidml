# simple execution example, do not merge

import numpy as np

import plaidml
import plaidml.exec as plaidml_exec
import plaidml.op as plaidml_op
from plaidml.edsl import *

def test_fake_quantize(I):
    O = plaidml_op.fake_quantize(I)
    return O

if __name__ == "__main__":
    input_shape = [32]
    a = np.array(np.random.uniform(0, 1, input_shape), dtype=np.float32)
    print("original", a)
    A = Tensor(LogicalShape(plaidml.DType.FLOAT32, input_shape))
    R = test_fake_quantize(A)
    program = Program('edsl_program', [R])
    binder = plaidml_exec.Binder(program)
    executable = binder.compile()
    binder.input(A).copy_from_ndarray(a)
    executable.run()
    plaidml_output = binder.output(R).as_ndarray()
    print("quantized:", plaidml_output)
