
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define GPUErrorAssertion(ans) {gpuAssert((ans), __FILE__, __LINE__);}

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, R"(GPUassert: $s $s $d
)", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void checkImage(unsigned int* img, unsigned int* result) {

    int blockSize = blockDim.x * blockDim.y;
    int tid = threadIdx.x + blockDim.y * threadIdx.y;
    int blockInGrid = blockIdx.x + gridDim.x * blockIdx.y;

    int idx = blockInGrid * blockSize + tid;

    result[idx] = (img[idx] <= 155) ? 0 : 255;

}

int main()
{
    // Image dimensions
    // Giving vaues for testing
    const int n = 100, m = 200;

    // Array size
    int m_size = sizeof(int) * n * m;

    int img_host[n * m];
    //unsigned int* img_host;
    unsigned int* result_host;

    unsigned int* img_device;
    unsigned int* result_device;

    //img_host = (unsigned int*)malloc(m_size);
    result_host = (unsigned int*)malloc(m_size);

    for (int i = 0; i < m * n; i++) {
        img_host[i] = i % 256;
    }

    dim3 block_size(8, 4, 1);
    dim3 grid_size(8, 4, 1);

    GPUErrorAssertion(cudaMalloc((void**)&img_device, m_size));
    GPUErrorAssertion(cudaMalloc((void**)&result_device, m_size));
    GPUErrorAssertion(cudaMemcpy(img_device, img_host, m_size, cudaMemcpyHostToDevice));

    checkImage <<<grid_size, block_size>>>  (img_device, result_device);
    GPUErrorAssertion(cudaDeviceSynchronize());

    GPUErrorAssertion(cudaMemcpy(result_host, result_device, m_size, cudaMemcpyDeviceToHost));

    // Print the result
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("|%d|", result_host[i * n + j]); // Access result from 1D array
        }
        printf("\n");
    }

    //free(img_host);
    free(result_host);
    GPUErrorAssertion(cudaFree(img_device));
    GPUErrorAssertion(cudaFree(result_device));

    return 0;

}
