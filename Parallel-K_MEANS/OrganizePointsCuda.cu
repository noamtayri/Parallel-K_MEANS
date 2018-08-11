
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Header.h"

#define NUM_THREADS_IN_BLOCK 1000

cudaError_t pointsOrgenaizeCuda(Cluster* clusters, Point *points, const int n, const int k, bool *flag);
cudaError_t Error(Point* dev_points, Cluster* dev_clusters, bool* dev_flags);

__device__ double distanceCuda(double x1, double y1, double x2, double y2)
{
	return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

__global__ void organizePointsKernel(Cluster *clusters, Point *points, bool *flags, const int k, const int n)
{
	double min = DBL_MAX;
	int minIdx;
	int i = blockIdx.x;
	int j = threadIdx.x;
	int idx = NUM_THREADS_IN_BLOCK * i + j;
	if (idx < n)
	{
		for (int l = 0; l < k; l++)
		{
			double tempDistance = distanceCuda(points[idx].x, points[idx].y, clusters[l].centerX, clusters[l].centerY);
			if (tempDistance < min) {
				minIdx = l;
				min = tempDistance;
			}
		}
		if (points[idx].myCluster != minIdx)
			flags[idx] = true;
		points[idx].myCluster = minIdx;
	}
}

int cudaOrganizePoints(Cluster* clusters, Point* points, int n, int k, bool *flag)
{
	*flag = false;

	// Add vectors in parallel.
	cudaError_t cudaStatus = pointsOrgenaizeCuda(clusters,points, n, k, flag);
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
cudaError_t pointsOrgenaizeCuda(Cluster* clusters, Point *points, const int n, const int k, bool *flag)
{
	Cluster *dev_clusters = 0;
	Point *dev_points = 0;
	cudaError_t cudaStatus;
	int numOfBlocks;
	if(n % NUM_THREADS_IN_BLOCK == 0)
		numOfBlocks = n / NUM_THREADS_IN_BLOCK;
	else
		numOfBlocks = n / NUM_THREADS_IN_BLOCK + 1;
	bool* dev_flags;
	bool* flags = (bool*)malloc(n * sizeof(bool));
	for (int i = 0; i < n; i++)
	{
		flags[i] = false;
	}
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		//goto Error;
		Error(dev_points, dev_clusters, dev_flags);
		return cudaStatus;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_points, n * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		Error(dev_points, dev_clusters, dev_flags);
		return cudaStatus;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_clusters, k * sizeof(Cluster));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		Error(dev_points, dev_clusters, dev_flags);
		return cudaStatus;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_flags, n * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		Error(dev_points, dev_clusters, dev_flags);
		return cudaStatus;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_flags, flags, n * sizeof(bool), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		Error(dev_points, dev_clusters, dev_flags);
		return cudaStatus;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_points, points, n * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		Error(dev_points, dev_clusters, dev_flags);
		return cudaStatus;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_clusters, clusters, k * sizeof(Cluster), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		Error(dev_points, dev_clusters, dev_flags);
		return cudaStatus;
	}

	// Launch a kernel on the GPU with one thread for each element.
	organizePointsKernel <<<numOfBlocks, NUM_THREADS_IN_BLOCK >>>(dev_clusters, dev_points, dev_flags, k,n);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		Error(dev_points, dev_clusters, dev_flags);
		return cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		Error(dev_points, dev_clusters, dev_flags);
		return cudaStatus;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(points, dev_points, n * sizeof(Point), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		Error(dev_points, dev_clusters, dev_flags);
		return cudaStatus;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(clusters, dev_clusters, k * sizeof(Cluster), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		Error(dev_points, dev_clusters, dev_flags);
		return cudaStatus;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(flags, dev_flags, n * sizeof(bool), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		Error(dev_points, dev_clusters, dev_flags);
		return cudaStatus;
	}

	for (int i = 0; i < n; i++)
	{
		if (flags[i] == true)
		{
			*flag = true;
			break;
		}
	}
	return Error(dev_points, dev_clusters, dev_flags);
}

cudaError_t Error(Point* dev_points, Cluster* dev_clusters, bool* dev_flags)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaFree(dev_points);
	if (cudaStatus != cudaSuccess) 
		fprintf(stderr, "cudaFree failed!");
	cudaStatus = cudaFree(dev_clusters);
	if (cudaStatus != cudaSuccess) 
		fprintf(stderr, "cudaFree failed!");
	cudaStatus = cudaFree(dev_flags);
	if (cudaStatus != cudaSuccess) 
		fprintf(stderr, "cudaFree failed!");

	return cudaStatus;
}
