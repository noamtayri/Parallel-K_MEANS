#include "Header.h"
#include <omp.h>

void initClusterArr(Cluster** clusters, Point* points, int k) 
{
#pragma omp parallel for
	for (int i = 0; i < k; i++) 
	{
		(*clusters)[i].id = i;
		(*clusters)[i].sumX = 0;
		(*clusters)[i].sumY = 0;
		(*clusters)[i].diameter = 0;
		(*clusters)[i].numOfPoints = 0;
		(*clusters)[i].centerX = points[i].x;
		(*clusters)[i].centerY = points[i].y;
	}
}

void newCicleClusterArr(Cluster** clusters, int k)
{
#pragma omp parallel for
	for (int i = 0; i < k; i++)
	{
		(*clusters)[i].id = i;
		(*clusters)[i].sumX = 0;
		(*clusters)[i].sumY = 0;
		(*clusters)[i].diameter = 0;
		(*clusters)[i].numOfPoints = 0;
	}
}

void recalculateClusterCenter(Cluster** clusters, int k)
{
#pragma omp parallel for
	for (int i = 0; i < k; i++) 
	{
		(*clusters)[i].centerX = (*clusters)[i].sumX / (*clusters)[i].numOfPoints;
		(*clusters)[i].centerY = (*clusters)[i].sumY / (*clusters)[i].numOfPoints;
		(*clusters)[i].sumX = 0;
		(*clusters)[i].sumY = 0;
		(*clusters)[i].diameter = 0;
		(*clusters)[i].numOfPoints = 0;
	}
}

double quality(Cluster* clusters, int k) 
{
	double sum = 0;
	int sumArrayLen = k < omp_get_max_threads() ? k : omp_get_max_threads();
	double* sumArr = (double*)malloc(sumArrayLen * sizeof(double));
	for (int i = 0; i < sumArrayLen; i++)
	{
		sumArr[i] = 0;
	}
#pragma omp parallel for
	for (int i = 0; i < k; i++) 
	{
		for (int j = 0; j < k; j++) 
		{
			if (j != i)
			{
				sumArr[omp_get_thread_num()] += clusters[i].diameter / calculateDistance(clusters[i].centerX, clusters[i].centerY, clusters[j].centerX, clusters[j].centerY);
			}
		}
	}
	for (int i = 0; i < sumArrayLen; i++)
	{
		sum += sumArr[i];
	}
	return sum / (k*(k - 1));
}

void diameterOMP(Cluster* clusters, Point* points, int n, int loadThreadBalance, int k)
{
	double* bigCluster = (double*)calloc(k*omp_get_max_threads(), sizeof(double));
	omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for
	for (int i = 0; i < loadThreadBalance; i++)
	{
		int idx = omp_get_thread_num();
		for (int j = i + 1; j < n; j++)
		{
			if (points[i].myCluster == points[j].myCluster)
			{
				double currentDiameter = calculateDistance(points[i].x, points[i].y, points[j].x, points[j].y);
				if (bigCluster[idx * k + points[i].myCluster] < currentDiameter)
					bigCluster[idx * k + points[i].myCluster] = currentDiameter;
			}
		}
	}

	//union the bigCluster arr
	for (int i = 0; i < k; i++)
	{
		for (int j = 0; j < omp_get_max_threads(); j++)
		{
			if (clusters[i].diameter < bigCluster[i + j*k])
				clusters[i].diameter = bigCluster[i + j*k];
		}
	}
}

void refreshClusterByPoints(Cluster* clusters, Point* points, int n, int k)
{
	Cluster* bigCluster = (Cluster*)malloc(k*omp_get_max_threads() * sizeof(Cluster));
	newCicleClusterArr(&bigCluster, k*omp_get_max_threads());
	omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		int idx = omp_get_thread_num();
		bigCluster[idx * k + points[i].myCluster].numOfPoints += 1;
		bigCluster[idx * k + points[i].myCluster].sumX += points[i].x;
		bigCluster[idx * k + points[i].myCluster].sumY += points[i].y;
	}

	//union the bigCluster arr
	for (int i = 0; i < k * omp_get_max_threads() ; i++)
	{
		clusters[i%k].numOfPoints += bigCluster[i].numOfPoints;
		clusters[i%k].sumX += bigCluster[i].sumX;
		clusters[i%k].sumY += bigCluster[i].sumY;
	}
}