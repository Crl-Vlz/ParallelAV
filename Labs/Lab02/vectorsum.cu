
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>

__global__ void sumVectors(int* result, const int* data_a, const int* data_b, const int* data_c) {
    int blockSize = blockDim.x * blockDim.y * blockDim.z;
    int tid = threadIdx.x + blockDim.y * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
    int blockInGrid = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;

    int idx = blockInGrid * blockSize + tid;
    result[idx] = data_a[idx] + data_b[idx] + data_c[idx];
    printf("%d + %d + %d = %d\n", data_a[idx], data_b[idx], data_c[idx], result[idx]);
}

int main()
{

    dim3 block_size(4, 4, 4);
    dim3 grid_size(3, 3, 3);

    std::random_device rd;
    std::mt19937 generator(rd());

    int* a_cpu;
    int* b_cpu;
    int* c_cpu;

    int* a_device;
    int* b_device;
    int* c_device;
    int* result_device;

    // Memory allocation data
    const int array_size = 10000;
    const int data_size = sizeof(int) * array_size;
    a_cpu = (int*)malloc(data_size);
    b_cpu = (int*)malloc(data_size);
    c_cpu = (int*)malloc(data_size);

    cudaMalloc((void**)&a_device, data_size);
    cudaMalloc((void**)&b_device, data_size);
    cudaMalloc((void**)&c_device, data_size);
    cudaMalloc((void**)&result_device, data_size);

    const int min_random = 1;
    const int max_random = 1000;


    std::uniform_int_distribution<int> distribution(min_random, max_random);

    for (int i = 0; i < array_size; i++) {
        a_cpu[i] = distribution(generator);
        b_cpu[i] = distribution(generator);
        c_cpu[i] = distribution(generator);
    }

    cudaMemcpy(a_device, a_cpu, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_cpu, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(c_device, c_cpu, data_size, cudaMemcpyHostToDevice);

    sumVectors << <grid_size, block_size >> > (result_device, a_device, b_device, c_device);

    cudaDeviceReset();
    cudaFree(c_device);
    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(result_device);

    free(a_cpu);
    free(b_cpu);
    free(c_cpu);

    return 0;
}
