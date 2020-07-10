import timeit

import numpy as np
import plaidml2 as plaidml
import plaidml2.edsl as edsl
import plaidml2.exec as plaidml_exec
import plaidml2.settings as plaidml_settings


def set_gpu():
    devices = sorted(plaidml_exec.list_devices())
    targets = sorted(plaidml_exec.list_targets())

    dev_idx = [idx for idx, s in enumerate(devices) if 'uhd_graphics' in s][0]
    tgt_idx = [idx for idx, s in enumerate(targets) if ('opencl' in s and 'intel' in s)][0]

    plaidml_settings.set('PLAIDML_DEVICE', devices[dev_idx])
    plaidml_settings.set('PLAIDML_TARGET', targets[tgt_idx])

    device = plaidml_settings.get('PLAIDML_DEVICE')
    target = plaidml_settings.get('PLAIDML_TARGET')

    print("Selected device, target:")
    print("{},  {}".format(device, target))

    plaidml_settings.save()


def set_cpu():
    devices = sorted(plaidml_exec.list_devices())
    targets = sorted(plaidml_exec.list_targets())

    dev_idx = [idx for idx, s in enumerate(devices) if 'cpu' in s][0]
    tgt_idx = [idx for idx, s in enumerate(targets) if 'cpu' in s][0]

    plaidml_settings.set('PLAIDML_DEVICE', devices[dev_idx])
    plaidml_settings.set('PLAIDML_TARGET', targets[tgt_idx])

    device = plaidml_settings.get('PLAIDML_DEVICE')
    target = plaidml_settings.get('PLAIDML_TARGET')

    print("Selected device, target:")
    print("{},  {}".format(device, target))

    plaidml_settings.save()


def pl_diff(I):
    M = edsl.TensorDim()
    m = edsl.TensorIndex()
    I.bind_dims(M)
    O = edsl.TensorOutput(M)
    I_neg = -I
    O[m] = I[m + 1] + I_neg[m]
    return O


def pl_matmul_2D(A, B):
    I, J, K = edsl.TensorDims(3)
    i, j, k = edsl.TensorIndexes(3)
    A.bind_dims(I, J)
    B.bind_dims(J, K)
    C = edsl.TensorOutput(I, K)
    C[i, k] += A[i, j] * B[j, k]
    return C


def run_program_1i_1o(I, I_data, O, benchmark=False):
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


def run_program_2i_1o(A, A_data, B, B_data, O, benchmark=False):
    program = edsl.Program('integral_program', [O])
    binder = plaidml_exec.Binder(program)
    executable = binder.compile()

    def run():
        binder.input(A).copy_from_ndarray(A_data)
        binder.input(B).copy_from_ndarray(B_data)
        executable.run()
        return binder.output(O).as_ndarray()

    if benchmark:
        # the first run will compile and run
        print('compiling...')
        result = run()

        # subsequent runs should not include compile time
        print('running...')
        ITERATIONS = 30
        elapsed = timeit.timeit(run, number=ITERATIONS)
        print('runtime:', elapsed / ITERATIONS)
    else:
        result = run()
    return result


def main():

    # Diff
    # N = 200000000
    # I_np = np.linspace(0, 1, N)

    # print("Testing CPU: ")
    # set_cpu()
    # I_pl = edsl.Tensor(edsl.LogicalShape(plaidml.DType.FLOAT32, I_np.shape))
    # O_pl = run_program_1i_1o(I_pl, I_np, pl_diff(I_pl), True)

    # print()
    # print("Testing GPU: ")
    # set_gpu()
    # I_pl = edsl.Tensor(edsl.LogicalShape(plaidml.DType.FLOAT32, I_np.shape))
    # O_pl = run_program_1i_1o(I_pl, I_np, pl_diff(I_pl), True)

    # Matmul
    N = 1000
    A_np = np.random.rand(N, N)
    B_np = np.random.rand(N, N)

    print("Testing CPU: ")
    set_cpu()
    A_pl = edsl.Tensor(edsl.LogicalShape(plaidml.DType.FLOAT32, A_np.shape))
    B_pl = edsl.Tensor(edsl.LogicalShape(plaidml.DType.FLOAT32, B_np.shape))
    O_pl = run_program_2i_1o(A_pl, A_np, B_pl, B_np, pl_matmul_2D(A_pl, B_pl), True)

    print()
    print("Testing GPU: ")
    set_gpu()
    A_pl = edsl.Tensor(edsl.LogicalShape(plaidml.DType.FLOAT32, A_np.shape))
    B_pl = edsl.Tensor(edsl.LogicalShape(plaidml.DType.FLOAT32, B_np.shape))
    O_pl = run_program_2i_1o(A_pl, A_np, B_pl, B_np, pl_matmul_2D(A_pl, B_pl), True)


if __name__ == '__main__':
    main()
