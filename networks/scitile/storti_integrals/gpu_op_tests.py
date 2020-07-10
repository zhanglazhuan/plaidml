import timeit

import numpy as np
import plaidml2 as plaidml
import plaidml2.edsl as edsl
import plaidml2.exec as plaidml_exec


def pl_diff(I):
    M = edsl.TensorDim()
    m = edsl.TensorIndex()
    I.bind_dims(M)
    O = edsl.TensorOutput(M)
    I_neg = -I
    O[m] = I[m + 1] + I_neg[m]
    return O


def run_program(I, I_data, O, benchmark=False):
    program = edsl.Program('integral_program', [O])
    binder = plaidml_exec.Binder(program)
    executable = binder.compile()

    def run():
        binder.input(I).copy_from_ndarray(I_data)
        executable.run()
        return binder.output(O).as_ndarray()

    if benchmark:
        # the first run will compile and run
        print('compiling...')
        result = run()

        # subsequent runs should not include compile time
        print('running...')
        ITERATIONS = 10
        elapsed = timeit.timeit(run, number=ITERATIONS)
        print('runtime:', elapsed / ITERATIONS)
    else:
        result = run()
    return result


def main():
    N = int(input("Enter N: "))
    I_np = np.linspace(0, 1, N)

    I_pl = edsl.Tensor(edsl.LogicalShape(plaidml.DType.FLOAT32, I_np.shape))
    O_pl = run_program(I_pl, I_np, pl_diff(I_pl), True)


if __name__ == '__main__':
    main()
