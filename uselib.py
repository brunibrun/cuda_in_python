import ctypes
import numpy as np


def setup_cuda_function():

    # load our shared library from current folder
    lib = np.ctypeslib.load_library('libaddvectors.so', '.')

    # define arguments and return type for the function (not strictly needed but helps with using correct types)
    double_1d = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1)
    lib.wrap_add_vectors.argtypes = [double_1d, double_1d, double_1d, ctypes.c_int]
    lib.wrap_add_vectors.restype = ctypes.c_void_p

    # return our function
    return lib.wrap_add_vectors


if __name__ == "__main__":

    # obtain our function from the shared library
    addvectors_cuda = setup_cuda_function()

    # initialize test data
    num_rows = 5
    a = np.random.randn(num_rows)
    b = np.random.randn(num_rows)

    # initialize output array
    out = np.empty(num_rows)

    # use our CUDA function!
    addvectors_cuda(a, b, out, num_rows)

    # print some results
    print("a: \n", a)
    print("b: \n", b)
    print("out: \n", out)
