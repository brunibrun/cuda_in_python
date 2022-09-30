#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>


#define CHECK(call)                                                \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
            exit(1);                                               \
        }                                                          \
    }


__global__ void add_vectors(double *a, double *b, double *out, const int num_rows)
{
    int idx;
    idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (; (idx < num_rows); idx += blockDim.x * gridDim.x)
        out[idx] = a[idx] + b[idx];
}


extern "C"
{
    void wrap_add_vectors(double *a, double *b, double *out, const int num_rows)
    {
        // alloc device memory
        size_t vec_bytes = num_rows * sizeof(double);

        double *a_device;
        double *b_device;
        double *out_device;

        CHECK(cudaMalloc(&a_device, vec_bytes));
        CHECK(cudaMalloc(&b_device, vec_bytes));
        CHECK(cudaMalloc(&out_device, vec_bytes));

        // copy data to device
        CHECK(cudaMemcpy(a_device, a, vec_bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(b_device, b, vec_bytes, cudaMemcpyHostToDevice));

        // launch kernel
        int threads_per_block = 256;
        int blocks_per_grid = (num_rows + threads_per_block - 1) / threads_per_block;

        add_vectors<<<threads_per_block, blocks_per_grid>>>(a_device, b_device, out_device, num_rows);

        CHECK(cudaDeviceSynchronize());

        // copy data back to host
        CHECK(cudaMemcpy(out, out_device, vec_bytes, cudaMemcpyDeviceToHost));

        // free device memory
        CHECK(cudaFree(a_device));
        CHECK(cudaFree(b_device));
        CHECK(cudaFree(out_device));

        CHECK(cudaDeviceReset());
    }
}