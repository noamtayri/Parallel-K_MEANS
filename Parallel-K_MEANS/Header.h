#define _CRT_SECURE_NO_DEPRECATE
#pragma warning (disable : 4996)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <mpi.h>
#include <string.h>

#define NO_SWITCH_POINTS 0
#define SWITCH_POINTS 1

struct Cluster {
	int id;
	int numOfPoints;
	double centerX;
	double centerY;
	double sumX;
	double sumY;
	double diameter;
};

struct Point {
	double x;
	double y;
	double vx;
	double vy;
	int myCluster;
};
//Methods.cpp methods
double calculateDistance(double x1, double y1, double x2, double y2);
void readFromFile(Point** points, int* n, int* k, double* t, double* dt, int* limit, double* qm);
void refreshPoints(Point** points, int start, int n, double t);
bool organizePoints(Cluster** clusters, Point** points, int n, int start, int k);
void calculateDiameter(Cluster** clusters, int k, Point* points, int amount, int n);
void writeToFile(Cluster* clusters, int k, double q, double t);

//CUDA methods
int cudaRefreshPoints(Point* points, int n, double t);
int cudaOrganizePoints(Cluster* clusters, Point* points, int n, int k, bool *flag);

//OpenMP methods
void initClusterArr(Cluster** clusters, Point* points, int k);
void newCicleClusterArr(Cluster** clusters, int k);
void recalculateClusterCenter(Cluster** clusters, int k);
double quality(Cluster* clusters, int k);
void diameterOMP(Cluster* clusters, Point* points, int n, int loadThreadBalance, int k);
void refreshClusterByPoints(Cluster* clusters, Point* points, int n, int k);