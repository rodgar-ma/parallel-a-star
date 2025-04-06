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
	char mssg[LENGTH], messages[MAX_NODES][LENGTH];
	
	MPI_Status  status;
	MPI_Request request, *p_requests;
	
	MPI_Init(&argc, &argv);
	MPI_Get_processor_name(name, &len);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	printf("< %s >: process %d of %d\n", name, rank, np);
	
	if (rank == 0) {
		p_requests = (MPI_Request *) malloc(sizeof(MPI_Request)*(np - 1));
		for(int i=1; i<np; i++) {
			MPI_Irecv(messages[i], LENGTH, MPI_CHAR, MPI_ANY_SOURCE, 10, MPI_COMM_WORLD, &p_requests[i-1]);
//			printf("After Irecv: message \"%s\" received\n", messages[i]);
		}
		printf("process %d: performing other tasks while receiving messages...\n", rank);
	}
	else {
		snprintf(mssg, LENGTH, "Hello from Process %d", rank);
		MPI_Isend(mssg, strlen(mssg) + 1, MPI_CHAR, 0, 10, MPI_COMM_WORLD, &request);
		printf("process %d sending message to process 0\n", rank); 
	}
	
	if (rank == 0) {
		for(int i=1; i<np; i++) 
		{
			MPI_Wait(&p_requests[i-1], &status);
			printf("process 0: message \"%s\" received!\n", messages[i]);
		}
		free(p_requests);
	}
	else {
		MPI_Wait(&request, &status);
	}
	
	MPI_Finalize();
	return(EXIT_SUCCESS);
}
