#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "wb.h"

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

#define wbCheck(stmt) \
 do { \
 cudaError_t err = stmt; \
 if (err != cudaSuccess) { \
 wbLog(ERROR, "Failed to run stmt ", #stmt); \
 wbLog(ERROR, "Got CUDA error ... ", cudaGetErrorString(err)); \
 return -1; \
 } \
 } while (0)

#define CLAMP(val, start, end) ((val) < (start) ? (start) : ((val) > (end) ? (end) : (val)))
#define value(arry, i, j, k) arry[(( i) * width + (j)) * depth + (k)]
#define in(i, j, k) value(input, i, j, k)
#define out(i, j, k) value(output, i, j, k)

__global__ void stencil(float *output, float *input, int width, int height, int depth) {
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Hello\n");
	if (x < width && y < height)
	{
		for (int k = 0; k < depth; k++)
		{
			float v = .5f;
			//float v = in(x, y, k + 1) + in(x, y, k - 1) + in(x, y + 1, k) + in(x, y - 1, k) + in(x + 1, y, k) + in(x - 1, y, k) - 6 * in(x, y, k);
			output[(y * width + x) * depth + k] = v;
			//out(x, y, k) = CLAMP(v, 0.0f, 1.0f);
		}
	}
}

static void launch_stencil(float *deviceOutputData, float *deviceInputData, int width, int height, int depth) {
	dim3 DimBlock(16, 16, 1);
	dim3 DimGrid((width-1) / DimBlock.x + 1, (height-1) / DimBlock.y + 1, 1);

	stencil KERNEL_ARGS2(DimGrid, DimBlock) (deviceInputData, deviceInputData, width, height, depth);
}

int main(int argc, char *argv[]) {
	wbArg_t arg;
	int width;
	int height;
	int depth;
	char *inputFile;
	wbImage_t input;
	wbImage_t output;
	float *hostInputData;
	float *hostOutputData;
	float *deviceInputData;
	float *deviceOutputData;
	arg = wbArg_read(argc, argv);
	inputFile = wbArg_getInputFile(arg, 0);
	input = wbImport(inputFile);
	width = wbImage_getWidth(input);
	height = wbImage_getHeight(input);
	depth = wbImage_getChannels(input);
	output = wbImage_new(width, height, depth);
	hostInputData = wbImage_getData(input);
	hostOutputData = wbImage_getData(output);
	wbTime_start(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **)&deviceInputData, width * height * depth * sizeof(float));
	cudaMalloc((void **)&deviceOutputData, width * height * depth * sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation");
	wbTime_start(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInputData, hostInputData, width * height * depth * sizeof(float), cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU");
	wbTime_start(Compute, "Doing the computation on the GPU");
	launch_stencil(deviceOutputData, deviceInputData, width, height, depth);
	wbTime_stop(Compute, "Doing the computation on the GPU");
	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputData, deviceOutputData, width * height * depth * sizeof(float), cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

	int i, j;
	FILE *fp = fopen("first.ppm", "wb"); /* b - binary mode */
	(void)fprintf(fp, "P6\n%d %d\n255\n", width, height);
	for (j = 0; j < height; ++j)
	{
		for (i = 0; i < width; ++i)
		{
			static unsigned char color[3];
			int offset = (i + j*width) * depth;

			color[0] = (char)(hostOutputData[offset] * 255.0f);  /* red */
			color[1] = (char)(hostOutputData[offset + 1] * 255.0f);  /* green */
			color[2] = (char)(hostOutputData[offset + 2] * 255.0f);  /* blue */

			//std::cout << "r:" << hostOutputImageData[offset]  << "g:" << hostOutputImageData[offset+1] << "b:" << hostOutputImageData[offset+2] << std::endl;
			(void)fwrite(color, 1, 3, fp);
		}
	}
	(void)fclose(fp);

	cudaFree(deviceInputData);
	cudaFree(deviceOutputData);
	wbImage_delete(output);
	wbImage_delete(input);
	return 0;
}

