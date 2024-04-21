#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__device__ void device_merge(int* sub_array, int* temp, int index, int size) {
    int index_left = index, index_right = index + size / 2;
    for (int i = 0; i < size; i++) {
        if (index_left < index + size / 2 && (index_right >= index + size || sub_array[index_left] < sub_array[index_right])) {
            temp[index + i] = sub_array[index_left];
            index_left++;
        }
        else {
            temp[index + i] = sub_array[index_right];
            index_right++;
        }
    }
    for (int i = 0; i < size; i++) sub_array[index + i] = temp[index + i];
}

__global__ void device_merge_sort(int* array, int* temp, int index, int size) {
    if (size == 1) return; // Stop at 1 element arrays
    int left_size = size / 2;
    int right_size = size - left_size;
    device_merge_sort << <1, 1 >> > (array, temp, index, left_size); // Left sub array
    device_merge_sort << <1, 1 >> > (array, temp, index + left_size, right_size); // Right sub array
    __syncthreads();
    device_merge(array, temp, index, size);
}

int main()
{
    const int arr_size = 16;
    const int data_size = sizeof(int) * arr_size;

    int* host_arr = (int*)malloc(data_size);
    int* dev_arr;
    int* dev_temp;
    cudaMalloc((void**)&dev_arr, data_size);
    cudaMalloc((void**)&dev_temp, data_size);

    for (int i = 0; i < arr_size; i++) host_arr[i] = rand();

    cudaMemcpy(dev_arr, host_arr, data_size, cudaMemcpyHostToDevice);

    device_merge_sort << <1, 1 >> > (dev_arr, dev_temp, 0, arr_size);

    cudaMemcpy(host_arr, dev_arr, data_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(dev_arr);
    cudaFree(dev_temp);

    for (int i = 0; i < arr_size; i++) {
        printf("%d ", host_arr[i]);
    }
    printf("\n");

    free(host_arr);

    return 0;
}
