#include "wb.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
void atomicAdd(void * adr, int rhs);
void __syncthreads();
#endif

#define NUM_BINS 256
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

__global__ void kernel_count(unsigned int *data, unsigned int input_length,unsigned int *output, unsigned int output_length)
{
	__shared__ unsigned int output_private[NUM_BINS];	

	int i = threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int stride_b = blockDim.x;
	while (i < NUM_BINS)
	{
		output_private[i] = 0;
		i += stride_b;
	}
	__syncthreads();

	i = threadIdx.x + blockIdx.x * blockDim.x;
	while (i < input_length)
	{
		atomicAdd(&(output_private[data[i]]), 1);
		
		i += stride;
	}

	__syncthreads();
	i = threadIdx.x;
	while (i < NUM_BINS)
	{
		atomicAdd(&(output[i]), output_private[i]);
		i += stride_b;
	}

}


int main(int argc, char *argv[]) {
	wbArg_t args;
	int inputLength;
	unsigned int *hostInput;
	unsigned int *hostBins;
	unsigned int *deviceInput;
	unsigned int *deviceBins;
	args = wbArg_read(argc, argv);
	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0), &inputLength, "Integer");
	hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
	wbTime_stop(Generic, "Importing data and creating memory on host");
	wbLog(TRACE, "The input length is ", inputLength);
	wbLog(TRACE, "The number of bins is ", NUM_BINS);
	wbTime_start(GPU, "Allocating GPU memory.");
	//@@ Allocate GPU memory here
	unsigned int * d_Bins = nullptr;
	unsigned int *d_Input = nullptr;
	cudaMalloc(&d_Input, inputLength * sizeof(unsigned int));
	cudaMalloc(&d_Bins, NUM_BINS * sizeof(unsigned int));

	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(GPU, "Allocating GPU memory.");
	wbTime_start(GPU, "Copying input memory to the GPU.");
	//@@ Copy memory to the GPU here
	cudaMemset(d_Bins, 0, sizeof(unsigned int) * NUM_BINS);
	cudaMemcpy(d_Input, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyKind::cudaMemcpyHostToDevice);

	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(GPU, "Copying input memory to the GPU.");
	// Launch kernel
	// ----------------------------------------------------------
	wbLog(TRACE, "Launching kernel");
	wbTime_start(Compute, "Performing CUDA computation");
	//@@ Perform kernel computation here

	int blockSize(256);
	int gridSize((inputLength - 1) / blockSize + 1);

	kernel_count KERNEL_ARGS2(gridSize, blockSize) (d_Input, inputLength, d_Bins, NUM_BINS);

	wbTime_stop(Compute, "Performing CUDA computation");
	wbTime_start(Copy, "Copying output memory to the CPU");

	//@@ Copy the GPU memory back to the CPU here
	cudaMemcpy(hostBins, d_Bins, NUM_BINS * sizeof(unsigned int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(Copy, "Copying output memory to the CPU");
	wbTime_start(GPU, "Freeing GPU Memory");
	//@@ Free the GPU memory here
	cudaFree(d_Bins);
	cudaFree(d_Input);
	wbTime_stop(GPU, "Freeing GPU Memory");

	for (int i = 0; i < NUM_BINS; i++)
	{
		std::cout << "Char : " << (char)i << "(" << (int) i << ") : " << hostBins[i] << std::endl;
	}

	free(hostBins);
	free(hostInput);
	return 0;
}
