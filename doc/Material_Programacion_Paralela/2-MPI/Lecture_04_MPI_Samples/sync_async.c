#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LENGTH   32
#define MAX_NODES 4

int main(int argc, char *argv[])
{
	int  np, rank, len;
	char name[MPI_MAX_PROCESSOR_NAME];
	char mssg[LENGTH];;
	
	MPI_Status  status;
	MPI_Request *p_requests;
	
	MPI_Init(&argc, &argv);
	MPI_Get_processor_name(name, &len);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	printf("< %s >: process %d of %d\n", name, rank, np);
	
	if (rank == 0) {
		p_requests = (MPI_Request *) malloc(sizeof(MPI_Request)*(np - 1));
		printf("process %d sending messages...\n", rank); 
     	for(int i=1; i<np; i++) {
			sprintf(mssg, "Hello Process %d", i);
			MPI_Isend(mssg, strlen(mssg) + 1, MPI_CHAR, i, 10, MPI_COMM_WORLD, &p_requests[i-1]);
		}
	}
	else {
		MPI_Recv(mssg, LENGTH, MPI_CHAR, 0, 10, MPI_COMM_WORLD, &status);
       	printf("process %d: \"%s\" received!\n", rank, mssg);
	}
	
	if (rank == 0) {
//		MPI_Waitall(np-1, p_requests, MPI_STATUSES_IGNORE);
		free(p_requests);
	}
	
	MPI_Finalize();
	return(EXIT_SUCCESS);
}