#include "Header.h"

int k_means(int argc, char *argv[]);

int main(int argc, char *argv[])
{
	return k_means(argc, argv);
}

//the k_means algorithm
int k_means(int argc, char *argv[])
{
	//===========================================MPI PROP=========================================================

	int myid, numprocs;
	MPI_Comm comm;
	MPI_Status status;
	struct Point point;
	struct Cluster cluster;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	if (numprocs != 3)
	{
		printf("need to be 3 proccess\n");
		fflush(stdout);
		MPI_Finalize();
		return 1;
	}
	MPI_Datatype PointMPIType;
	MPI_Datatype type[5] = { MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_INT };
	int blocklen[5] = { 1,1,1,1,1 };
	MPI_Aint disp[5];

	disp[0] = (char *)&point.x - (char *)&point;
	disp[1] = (char *)&point.y - (char *)&point;
	disp[2] = (char *)&point.vx - (char *)&point;
	disp[3] = (char *)&point.vy - (char *)&point;
	disp[4] = (char *)&point.myCluster - (char *)&point;

	MPI_Type_create_struct(5, blocklen, disp, type, &PointMPIType);
	MPI_Type_commit(&PointMPIType);


	MPI_Datatype ClusterMPIType;
	MPI_Datatype type2[7] = { MPI_INT,MPI_INT,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE };
	int blocklen2[7] = { 1,1,1,1,1,1,1 };
	MPI_Aint disp2[7];

	disp2[0] = (char *)&cluster.id - (char *)&cluster;
	disp2[1] = (char *)&cluster.numOfPoints - (char *)&cluster;
	disp2[2] = (char *)&cluster.centerX - (char *)&cluster;
	disp2[3] = (char *)&cluster.centerY - (char *)&cluster;
	disp2[4] = (char *)&cluster.sumX - (char *)&cluster;
	disp2[5] = (char *)&cluster.sumY - (char *)&cluster;
	disp2[6] = (char *)&cluster.diameter - (char *)&cluster;

	MPI_Type_create_struct(7, blocklen2, disp2, type2, &ClusterMPIType);
	MPI_Type_commit(&ClusterMPIType);
	
	//===========================================PROPERTIES===========================================================


	clock_t start = clock();
	int N, K, LIMIT;
	double dT, QM, T;
	Point* points;
	Point* masterPoints;
	Cluster* clusters;
	//prop for mpi
	Cluster* bigClusters;
	double information[6];
	bool* allFlags = (bool*)malloc((numprocs + 1) * sizeof(bool));
	int* integerallFlags = (int*)malloc(numprocs * sizeof(int));
	bool* masterFlags = (bool*)malloc((numprocs + 1) * sizeof(bool));
	int* integerMasterFlags = (int*)malloc((numprocs + 1) * sizeof(int));

	//===========================================Algorithm Body=========================================================

	//id 0 read from file and brodcast prop and points
	if (myid == 0)
	{
		masterPoints = (Point*)malloc(N * sizeof(Point));
		readFromFile(&points, &N, &K, &T, &dT, &LIMIT, &QM);
		information[0] = N;
		information[1] = K;
		information[2] = T;
		information[3] = dT;
		information[4] = LIMIT;
		information[5] = QM;
		readFromFile(&masterPoints, &N, &K, &T, &dT, &LIMIT, &QM);
	}
	MPI_Bcast(information, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	N = (int)information[0];
	K = (int)information[1];
	T = information[2];
	dT = information[3];
	LIMIT = (int)information[4];
	QM = information[5];
	if(myid > 0)
		points = (Point*)malloc(N * sizeof(Point));
	//send all the points
	MPI_Bcast(points, N, PointMPIType, 0, MPI_COMM_WORLD);
	//everybody init clusters array
	clusters = (Cluster*)malloc(K * (sizeof(Cluster)));
	bigClusters = (Cluster*)malloc(K*numprocs * (sizeof(Cluster)));
	for (int i = 0; i < K*numprocs; i++)
	{
		bigClusters[i].numOfPoints = 0;
		bigClusters[i].sumX = 0;
		bigClusters[i].sumY = 0;
	}
	initClusterArr(&clusters, points, K);
	int reminder = N % numprocs;
	int partition = N / numprocs;
	//calculate diameter partition of work
	int sumOfNumOfProc = numprocs * (numprocs + 1) / 2;
	int part = N / sumOfNumOfProc;
	int* startPointsForEachProcDiameter = (int*)malloc(numprocs * sizeof(int));
	startPointsForEachProcDiameter[0] = 0;
	for (int i = 1; i <= numprocs; i++)
	{
		startPointsForEachProcDiameter[i] = startPointsForEachProcDiameter[i - 1] + i * part;
	}

	//start of k_means algorithm
	//k_means algorithm outer loop
	for (double i = 0; i < T; i += dT)
	{
		if (myid == 0)
		{
			printf("%f\n", i);
			fflush(stdout);
		}
		if (i != 0)
		{
			//===================================================refresh points===============================================
			cudaRefreshPoints(points + (myid * partition), partition, dT);
			if (myid == 0)
				if (reminder != 0)
				{
					refreshPoints(&points, N - reminder, N, dT);

					for (int z = N - reminder; z < N; z++)
					{
						masterPoints[z] = points[z];
					}
				}
			MPI_Gather(points + (myid * partition), N / numprocs, PointMPIType, masterPoints, N / numprocs, PointMPIType, 0, MPI_COMM_WORLD);
			if (myid == 0)
				memcpy(points, masterPoints, N * sizeof(Point));
			MPI_Bcast(points, N, PointMPIType, 0, MPI_COMM_WORLD);
		}
		newCicleClusterArr(&clusters, K);
		
		//k_means algorithm inner loop
		for (int j = 0; j < LIMIT; j++)
		{
			//=========================================organize points==============================================
			if (myid == 0)
			{
				printf("%d\n", j);
				fflush(stdout);
			}
			bool flag = false;
			int intFlag = 0;
			bool tempFlag = false;
			cudaOrganizePoints(clusters, points + myid*partition, N/numprocs, K, &tempFlag);
			if (tempFlag == true)
				integerallFlags[myid] = 1;
			else
				integerallFlags[myid] = 0;

			// handle in the reminder
			if (myid == 0)
			{
				if (reminder != 0)
				{
					masterFlags[numprocs] = organizePoints(&clusters, &points, N, N-reminder, K);
					
					for (int z = N - reminder; z < N; z++)
					{
						masterPoints[z] = points[z];
					}
				}
			}

			MPI_Gather(points + (myid * partition), N / numprocs, PointMPIType, masterPoints, N / numprocs, PointMPIType, 0, MPI_COMM_WORLD);
			if(myid == 0)
				memcpy(points, masterPoints, N * sizeof(Point));

			if (myid == 0)
				refreshClusterByPoints(clusters, masterPoints, N, K);
			
			//send all the flags
			MPI_Gather(integerallFlags + myid, 1, MPI_INT, integerMasterFlags, 1, MPI_INT, 0, MPI_COMM_WORLD);

			
			if (myid == 0)
			{
				for (int z = 0; z < numprocs; z++)
				{
					if(integerMasterFlags[z] == 1)
						masterFlags[z] = true;
					else
						masterFlags[z] = false;
				}
				for (int z = 0; z <= numprocs; z++)
				{
					if (masterFlags[z] == true)
					{
						flag = true;
						intFlag = 1;
						break;
					}
				}
			}

			//send the update clusters
			MPI_Bcast(clusters, K, ClusterMPIType, 0, MPI_COMM_WORLD);
			//send flag
			MPI_Bcast(&intFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
			if (myid > 0)
			{
				if (intFlag == 0)
					flag = false;
				else
					flag = true;
			}
			//send all points
			MPI_Bcast(points, N, PointMPIType, 0, MPI_COMM_WORLD);

			//=========================================================================================================
			recalculateClusterCenter(&clusters, K);

			if (flag == false) {
				break;
			}
		}

		//================================================Diameter======================================================
		diameterOMP(clusters, points + startPointsForEachProcDiameter[myid], N - startPointsForEachProcDiameter[myid], part * (myid + 1), K);
		if (myid == 0)
		{
			calculateDiameter(&clusters, K, points + (part * sumOfNumOfProc), N - (part * sumOfNumOfProc), N - (part * sumOfNumOfProc));
		}
		//recive clusters
		MPI_Gather(clusters, K, ClusterMPIType, bigClusters, K, ClusterMPIType, 0, MPI_COMM_WORLD);

		if (myid == 0)
		{
			for (int z = 0; z < K*numprocs; z++)
			{
				if (clusters[z%K].diameter < bigClusters[z].diameter)
					clusters[z%K].diameter = bigClusters[z].diameter;
			}
		}

		//send the update clusters
		MPI_Bcast(clusters, K, ClusterMPIType, 0, MPI_COMM_WORLD);
		
		//measure quality
		double q = quality(clusters, K);
		if (myid == 0)
		{
			printf("q = %f\n", q);
			fflush(stdout);
		}
		if (q <= QM) {
			if (myid == 0)
			{
				printf("First occurrence at t = %f with q = %f\n", i, q);
				printf("Centers of the clusters:\n");
				fflush(stdout);
				for (int i = 0; i < K; i++) 
				{
					printf("%f    %f\n", clusters[i].centerX, clusters[i].centerY);
					fflush(stdout);
				}
				writeToFile(clusters, K, q, i);
			}
			break;
		}
	}
	free(points);
	free(clusters);
	free(allFlags);
	free(startPointsForEachProcDiameter);
	free(masterFlags);
	free(bigClusters);
	if(myid == 0)
		free(masterPoints);
	free(integerMasterFlags);
	free(integerallFlags);
	clock_t end = clock();
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	if (myid == 0)
	{
		printf("time = %f\n", seconds);
		fflush(stdout);
	}
	MPI_Finalize();
	return 0;
}