# cuda_in_python

This repo contains a simple example on how to interface CUDA with Python using the 'ctypes'-package.

The example consists of a CUDA vector-addition function used in a Python script.

There are of course many more ways to interface CUDA and Python, but I was surprised by how simple this approach is and how well it works.
Another advantage of this approach is that the 'ctypes' package is included in the Python standard libraries, so there is very good documentation and support.

## How to run the example?
Compile the CUDA example code:

    nvcc addvectors.cu -Xcompiler -fPIC -shared -o libaddvectors.so

and then just run the Python file.

## Why use CUDA in Python?
This could be useful, for example, if you want to run parts of your code directly on the GPU or if you want to use existing CUDA libraries in Python.

## Some notes
- Using the 'ctypes' package allows (among other things) calling functions from shared libraries written in C or any other language that can export a C-compatible API (like CUDA or C++). To interface with CUDA, we thus need to wrap all the functions we want to use within Python in an *extern C* block. So we actually interface CUDA to C first, and then C to Python.

- You can also create a prettier interface using header files, as shown for example in https://github.com/chcomin/ctypes-numpy-example.
This would also be an option for using shared libraries where you can't add an *extern C* in the code.

- In the example I also use numpy's ctypeslib to make it easier to handle numpy arrays. But this is of course optional.