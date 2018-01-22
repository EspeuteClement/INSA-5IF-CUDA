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

#define cudaCheckError() {                                          \
cudaError_t e=cudaGetLastError();                                 \
  if(e!=cudaSuccess) {                                              \
    printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
    exit(0); \
  }                                                                 \
}

#define BLUR_SIZE 25

//@@ INSERT CODE HERE
__global__ void doBlur(float * in, float * out, int width, int height, int depth)
{
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < width && y < height)
	{
		long int counter = 0;
		float rgbValue[3] = { 0.0f,0.0f,0.0f };

		for (int oy = -BLUR_SIZE; oy <= BLUR_SIZE; oy++)
		{
			for (int ox = -BLUR_SIZE; ox <= BLUR_SIZE; ox++)
			{
				int new_x = x + ox;
				int new_y = y + oy;

				if (new_x >= 0 && new_x < width && new_y >= 0 && new_y < height)
				{
					int offset = (new_x + new_y * width) * depth;
					counter++;
					rgbValue[0] += in[offset];
					rgbValue[1] += in[offset + 1];
					rgbValue[2] += in[offset + 2];
				}
			}
	  }

	  int offset = (x + y * width) * depth;

	  rgbValue[0] /= counter;
	  rgbValue[1] /= counter;
	  rgbValue[2] /= counter;

	  out[offset] = rgbValue[0];
	  out[offset + 1] = rgbValue[1];
	  out[offset + 2] = rgbValue[2];

	  //printf(" r:%f, g:%f, b:%f\n", out[offset], out[offset + 1], out[offset + 2]);
	}
}

__global__ void print(const float const * data, int width, int height, int depth)
{
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x < width && y < height)
  {
    int offset = (x + y * width) * depth;
	  //printf(" r:%f, g:%f, b:%f\n", data[offset], data[offset + 1], data[offset + 2]);
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
	int imageChannels;
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
	imageChannels = inputImage.channels;
  outputImage = wbImage_new(imageWidth, imageHeight, 3);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");
  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
  imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData,
  imageWidth * imageHeight * imageChannels * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");
  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData, hostInputImageData,
  imageWidth * imageHeight * imageChannels * sizeof(float),
  cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");
  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");
	cudaDeviceSynchronize();

	dim3 DimBlock(16, 16, 1);
	dim3 DimGrid((imageWidth-1) / DimBlock.x + 1, (imageHeight-1) / DimBlock.y + 1, 1);
	//dim3 DimGrid(16, 16,1);
	doBlur KERNEL_ARGS2(DimGrid, DimBlock) (deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight, inputImage.channels);
	//print KERNEL_ARGS2(DimGrid, DimBlock) (deviceOutputImageData, imageWidth, imageHeight, inputImage.channels);
  wbTime_stop(Compute, "Doing the computation on the GPU");
  ///////////////////////////////////////////////////////
  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
  imageWidth * imageHeight * imageChannels * sizeof(float),
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
      static unsigned char color[3];
			int offset = (i + j*imageWidth) * imageChannels;

      color[0] = (char) (hostOutputImageData[offset] * 255.0f);  /* red */
      color[1] = (char) (hostOutputImageData[offset + 1] * 255.0f);  /* green */
      color[2] = (char) (hostOutputImageData[offset + 2] * 255.0f);  /* blue */
		
			//std::cout << "r:" << hostOutputImageData[offset]  << "g:" << hostOutputImageData[offset+1] << "b:" << hostOutputImageData[offset+2] << std::endl;
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
