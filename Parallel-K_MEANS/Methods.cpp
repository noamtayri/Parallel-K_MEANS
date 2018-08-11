#include "Header.h"

double calculateDistance(double x1, double y1, double x2, double y2) {
	return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

void readFromFile(Point** points, int* n, int* k, double* t, double* dt, int* limit, double* qm) {
	//FILE* f = fopen("input.txt", "r");
	FILE* f = fopen("D:\\Final_K_Means_NoamTayri\\Parallel-K_MEANS\\Parallel-K_MEANS\\input.txt", "r");
	int offset;
	offset = fscanf(f, "%d", n); //N - number of points
	offset = fscanf(f, "%d", k); //K - number of clusters to find
	offset = fscanf(f, "%lf", t); //T – defines the end of time interval [0, T]
	offset = fscanf(f, "%lf", dt); //dT – defines moments t = n*dT, n = { 0, 1, 2, … , T/dT} for which calculate the clusters and the quality
	offset = fscanf(f, "%d", limit); //LIMIT – the maximum number of iterations for K-MEAN algorithm. 
	offset = fscanf(f, "%lf", qm); //QM – quality measure to stop
	*points = (Point*)malloc(*n * sizeof(Point));
	for (int i = 0; i < *n; i++) {
		offset = fscanf(f, "%lf", &((*points)[i].x));
		offset = fscanf(f, "%lf", &((*points)[i].y));
		offset = fscanf(f, "%lf", &((*points)[i].vx));
		offset = fscanf(f, "%lf", &((*points)[i].vy));
	}
	fclose(f);
}

bool organizePoints(Cluster** clusters, Point** points, int n,int start, int k) {
	double min = DBL_MAX;
	int idx;
	bool isChange = false;
	for (int i = start; i < n; i++) {
		for (int j = 0; j < k; j++) {
			double tempDistance = calculateDistance((*points)[i].x, (*points)[i].y, (*clusters)[j].centerX, (*clusters)[j].centerY);
			if (tempDistance < min) {
				idx = j;
				min = tempDistance;
			}
		}
		if ((*points)[i].myCluster != idx)
			isChange = true;
		(*points)[i].myCluster = idx;
		min = DBL_MAX;
	}
	return isChange;
}

void calculateDiameter(Cluster** clusters, int k, Point* points, int amount, int n) {
	for (int i = 0; i < amount; i++) {
		for (int j = i + 1; j < n; j++) {
			if (points[i].myCluster == points[j].myCluster) {
				double currentDiameter = calculateDistance(points[i].x, points[i].y, points[j].x, points[j].y);
				if ((*clusters)[points[i].myCluster].diameter < currentDiameter) {
					(*clusters)[points[i].myCluster].diameter = currentDiameter;
				}
			}
		}
	}
}

void refreshPoints(Point** points, int start, int n, double t) {
	for (int i = start; i < n; i++) {
		(*points)[i].x = (*points)[i].x + t*(*points)[i].vx;
		(*points)[i].y = (*points)[i].y + t*(*points)[i].vy;
	}
}

void writeToFile(Cluster* clusters, int k, double q, double t) {
	FILE* f = fopen("D:\\Final_K_Means_NoamTayri\\Parallel-K_MEANS\\Parallel-K_MEANS\\output.txt", "w");
	fprintf(f,"First occurrence at t = %f with q = %f\n", t, q);
	fprintf(f,"Centers of the clusters:\n");
	for (int i = 0; i < k; i++)
	{
		fprintf(f,"%f    %f\n", clusters[i].centerX, clusters[i].centerY);
	}
}