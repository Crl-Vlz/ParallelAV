
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void unique_id_calculation_2d_2d(int * data)
{
    int tid = blockDim.x * threadIdx.x + threadIdx.y;
    
    int num_threads_in_a_block = blockDim.x * blockDim.y;
    int block_offset = blockIdx.x * num_threads_in_a_block;
    
    int num_threads_in_a_row = num_threads_in_a_block * gridDim.x;
    int row_offset = num_threads_in_a_row * blockIdx.y;

    int gid = tid + block_offset + row_offset;

    printf("blockIdx.x : %d, blockIdx.y %d, threadIdx.x : %d, gid : %d - data %d\n", blockIdx.x, blockIdx.y, tid, gid, data[gid]);
}

__global__ void uid22(int* data)
{
    int blockSize = blockDim.x * blockDim.y;
    int tid = threadIdx.x + blockDim.y * threadIdx.y;
    int blockInGrid = blockIdx.x + gridDim.x * blockIdx.y;

    int uid = blockInGrid * blockSize + tid;
    printf("blockIdx.x : %d, blockIdx.y %d, threadIdx.x : %d, gid : %d - data %d\n", blockIdx.x, blockIdx.y, tid, uid, data[uid]);
}

__global__ void uid33(int * data)
{
    int blockSize = blockDim.x * blockDim.y * blockDim.z;
    int tid = threadIdx.x + blockDim.y * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
    int blockInGrid = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;

    int uid = blockInGrid * blockSize + tid;
    printf("blockIdx.x : %d, blockIdx.y %d, blockIdx.z : %d, threadIdx.x : %d, threadIdx.y : %d, threadIdx.z : %d, gid : %d - data %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, uid, data[uid]);
}

int main()
{

    /*int array_size = 16;
    int array_byte_size = sizeof(int) * array_size;
    int h_data[] = { 23, 9, 4, 53, 65, 12, 1, 33, 22, 43, 56, 4, 76, 81, 94, 32 };*/
    int array_size = 64;
    int array_byte_size = sizeof(int) * array_size;
    int h_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64};

    int* d_data;
    cudaMalloc((void**)&d_data, array_byte_size);
    cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

    dim3 block(2, 2, 2);
    dim3 grid(2, 2, 2);

    /*unique_id_calculation_2d_2d<<<grid, block>>>(d_data);
    uid22 << <grid, block >> > (d_data);*/
    uid33 << <grid, block >> > (d_data);

    return 0;
}
