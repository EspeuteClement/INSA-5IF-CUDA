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
#endif


__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	printf("AM DOING STUFF");

	if (i < len)
	{
		out[i] = in1[i] + in2[i];
	}
}

int main(int argc, char **argv) {
    wbArg_t args;
    int inputLength;
    float *hostInput1;
    float *hostInput2;
	float *hostOutput;
    args = wbArg_read(argc, argv);
    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 =
    (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 =
    (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *)malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");
    wbLog(TRACE, "The input length is ", inputLength);
    wbTime_start(GPU, "Allocating GPU memory.");
	
	float * memInput1 = nullptr;
	cudaMalloc(&memInput1, inputLength * sizeof(float));

	float * memInput2 = nullptr;
	cudaMalloc(&memInput2, inputLength * sizeof(float));

	float * memOutput = nullptr;
	cudaMalloc(&memOutput, inputLength * sizeof(float));

    wbTime_stop(GPU, "Allocating GPU memory.");
    wbTime_start(GPU, "Copying input memory to the GPU.");

	cudaMemcpy(memInput1, hostInput1, inputLength * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(memInput2, hostInput2, inputLength * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
	//cudaMemset(memOutput, 0, inputLength * sizeof(float));

	//@@ Copy memory to the GPU here
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    //@@ Initialize the grid and block dimensions here
	
	int blockSize(256);
	int gridSize((inputLength-1)/blockSize + 1);

    wbTime_start(Compute, "Performing CUDA computation");
	


	vecAdd KERNEL_ARGS2(gridSize, blockSize) (memInput1, memInput2, memOutput, inputLength);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    wbTime_start(Copy, "Copying output memory to the CPU");

	cudaMemcpy(hostOutput, memOutput, inputLength * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    //@@ Copy the GPU memory back to the CPU here
    wbTime_stop(Copy, "Copying output memory to the CPU");
    wbTime_start(GPU, "Freeing GPU Memory");
	cudaFree(memInput1);
	cudaFree(memInput2);
	cudaFree(memOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, inputLength);
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

	getchar();

    return 0;
}
