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


#define wbCheck(stmt)\
    do {\
        cudaError_t err = stmt;\
        if (err != cudaSuccess) {\
            wbLog(ERROR, "Failed to run stmt ", #stmt);\
            wbLog(ERROR, "Got CUDA error ... ", cudaGetErrorString(err));\
            return -1;\
        }\
    } while (0)\
    
//@@ INSERT CODE HERE
__global__ void toGray(float *in, float *out, int w, int h, int channels)
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int index = y*w + x;

	if (x < w && y < h)
	{
		int rgbOffset = index * channels;
		out[index] = in[rgbOffset] * 0.21 + in[rgbOffset+1] * 0.71 + in[rgbOffset+2] * 0.07;
	}
}


int main(int argc, char *argv[]) {
    wbArg_t args;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char *inputImageFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float *hostInputImageData;
    float *hostOutputImageData;
    float *deviceInputImageData;
    float *deviceOutputImageData;
    args = wbArg_read(argc, argv); /* parse the input arguments */
    inputImageFile = wbArg_getInputFile(args, 0);
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    // For this lab the value is always 3
    imageChannels = wbImage_getChannels(inputImage);
    // Since the image is monochromatic, it only contains one channel
    outputImage = wbImage_new(imageWidth, imageHeight, 1);
    hostInputImageData = wbImage_getData(inputImage);



    hostOutputImageData = wbImage_getData(outputImage);
    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");
    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **)&deviceInputImageData,
    imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **)&deviceOutputImageData,
    imageWidth * imageHeight * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");
    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData, hostInputImageData,
    imageWidth * imageHeight * imageChannels * sizeof(float),
    cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");
    ///////////////////////////////////////////////////////

	dim3 DimBlock(16, 16, 1);
	dim3 DimGrid(imageWidth / DimBlock.x + 1, imageHeight / DimBlock.y + 1, 1);
    wbTime_start(Compute, "Doing the computation on the GPU");
    
	toGray KERNEL_ARGS2(DimGrid, DimBlock) (deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight, imageChannels);

    wbTime_stop(Compute, "Doing the computation on the GPU");
    ///////////////////////////////////////////////////////
    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData, deviceOutputImageData,
    imageWidth * imageHeight * sizeof(float),
    cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");
    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");
    
    
    int i, j;
    FILE *fp = fopen("first.ppm", "wb"); /* b - binary mode */
    (void) fprintf(fp, "P6\n%d %d\n255\n", imageWidth, imageHeight);
    for (j = 0; j < imageHeight; ++j)
    {
        for (i = 0; i < imageWidth; ++i)
        {
			static unsigned char color[6] = { 0 };
			unsigned char v = (unsigned char)(hostOutputImageData[j*imageWidth + i] * 255.0f);
            color[0] = v;  /* red */
            color[1] = v;  /* green */
            color[2] = v;  /* blue */
      
            (void) fwrite(color, 1, 3, fp);
        }
    }
    (void) fclose(fp);
    
    
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);
    return 0;
}
