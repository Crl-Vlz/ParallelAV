#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include "stdlib.h"
#include <iostream>

int main() {
	int size = 1 << 25;
	int bytes = size * sizeof(float);

	// Allocate the host memory
	float* h_a = (float *)malloc(bytes);

	// float* h_a;
	// cudaMallocHost((float**)&h_a, bytes);

	// Allocate the device memory
	float* d_a;
	cudaMalloc((float**)&d_a, bytes);

	// initialize the host memory
	for (int i = 0; i < size; i++) {
		h_a[i] = rand() % 10;
	}
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

	cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);

	// Free memory
	cudaFree(d_a);
	free(h_a);
	// cudaFreeHost(h_a);


}
