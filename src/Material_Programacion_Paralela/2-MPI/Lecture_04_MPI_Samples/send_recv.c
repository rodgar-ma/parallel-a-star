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
	
	MPI_Init(&argc, &argv);
	MPI_Get_processor_name(name, &len);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	printf("< %s >: process %d of %d\n", name, rank, np);
	
	if (rank > 0) {
		snprintf(mssg, LENGTH, "Hello from Process %d", rank);
		MPI_Send(mssg, strlen(mssg) + 1, MPI_CHAR, 0, 10, MPI_COMM_WORLD);
		printf("process %d sending message to process 0\n", rank); 
	}
	else {
     	for(int i=1; i<np; i++) {
			MPI_Recv(messages[i], LENGTH, MPI_CHAR, MPI_ANY_SOURCE, 10, MPI_COMM_WORLD, &status);
			printf("process 0 receiving message \"%s\"\n", messages[i]);
        }
	}
	
	MPI_Finalize();
	return(EXIT_SUCCESS);
}
