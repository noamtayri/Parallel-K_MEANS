
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Header.h"

#define NUM_THREADS_IN_BLOCK 1000

cudaError_t refreshPointsInitCuda(Point *points, const int n, const double t);
cudaError_t Error(Point* dev_points);

__global__ void refreshPointsKernel(Point *points, const double t)
{
	int i = blockIdx.x;
    int j = threadIdx.x;
	points[NUM_THREADS_IN_BLOCK*i + j].x = points[NUM_THREADS_IN_BLOCK*i + j].x + t*points[NUM_THREADS_IN_BLOCK*i + j].vx;
	points[NUM_THREADS_IN_BLOCK*i + j].y = points[NUM_THREADS_IN_BLOCK*i + j].y + t*points[NUM_THREADS_IN_BLOCK*i + j].vy;
}

int cudaRefreshPoints(Point* points, int n, double dt)
{

    // Add vectors in parallel.
    cudaError_t cudaStatus = refreshPointsInitCuda(points, n, dt);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t refreshPointsInitCuda(Point *points, const int n, const double t)
{
    Point *dev_points = 0;
    cudaError_t cudaStatus;

	int numOfBlocks;
	if (n % NUM_THREADS_IN_BLOCK == 0)
		numOfBlocks = n / NUM_THREADS_IN_BLOCK;
	else
		numOfBlocks = n / NUM_THREADS_IN_BLOCK + 1;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		Error(dev_points);
		return cudaStatus;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_points, n * sizeof(Point));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
		Error(dev_points);
		return cudaStatus;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_points, points, n * sizeof(Point), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
		Error(dev_points);
		return cudaStatus;
    }

    // Launch a kernel on the GPU with one thread for each element.
	refreshPointsKernel <<<numOfBlocks, NUM_THREADS_IN_BLOCK>>>(dev_points, t);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		Error(dev_points);
		return cudaStatus;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		Error(dev_points);
		return cudaStatus;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(points, dev_points, n * sizeof(Point), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
		Error(dev_points);
		return cudaStatus;
    }
	return Error(dev_points);
}

cudaError_t Error(Point* dev_points)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaFree(dev_points);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaFree failed!");
	
	return cudaStatus;
}