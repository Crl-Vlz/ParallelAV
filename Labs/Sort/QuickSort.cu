
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <ctime>

#define MAX_THREADS 128
#define N 512

void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high]; // pivot element (commonly last element)
    int i = (low - 1); // index of smaller element

    for (int j = low; j <= high - 1; j++) {
        // If current element is smaller than the pivot
        if (arr[j] < pivot) {
            i++; // increment index of smaller element
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]); // place pivot at the correct position
    return (i + 1);
}

void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high); // partition index

        // Recursively sort elements before and after partition
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

__device__ void device_swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

__device__ int device_partition(int* a, int low, int high) {
    int pivot = a[high]; // pivot element (commonly last element)
    int i = (low - 1); // index of smaller element

    for (int j = low; j <= high - 1; j++) {
        // If current element is smaller than the pivot
        if (a[j] < pivot) {
            i++; // increment index of smaller element
            device_swap(&a[i], &a[j]);
        }
    }
    device_swap(&a[i + 1], &a[high]); // place pivot at the correct position
    return (i + 1);
}

__global__ void device_quickSort(int * a, int low, int high) {
    if (low < high) {
        int index = device_partition(a, low, high);

        device_quickSort << <1, 1 >> > (a, low, index - 1);
        device_quickSort << <1, 1 >> > (a, index + 1, high);

    }
}

void printArray(int arr[], int n) {
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

int main()
{
    clock_t start, end;
    double cpu_time_used;

    int arr[] = { 10, 7, 8, 9, 1, 5, 23, 4, 92, 103, 23, 45, 29, 22, 101 };
    int n = sizeof(arr) / sizeof(arr[0]);
    int* a;
    int t = 100;
    int data_size = sizeof(int) * t;

    int* d_a;


    a = (int *)malloc(data_size);
    cudaMalloc((void**)&d_a, data_size);

    for (int i = 0; i < t; i++) {
        a[i] = rand();
    }

    cudaMemcpy(d_a, a, data_size, cudaMemcpyHostToDevice);

    start = clock();

    //quickSort(a, 0, t - 1);
    device_quickSort << < 1, 1 >> > (d_a, 0, t - 1);
    cudaDeviceSynchronize();
    end = clock();
    cudaMemcpy(a, d_a, data_size, cudaMemcpyDeviceToHost);
    printArray(a, t);

    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Clock time used CPU: %lf\n", cpu_time_used);

    cudaFree(d_a);
    free(a);

    return 0;
}
