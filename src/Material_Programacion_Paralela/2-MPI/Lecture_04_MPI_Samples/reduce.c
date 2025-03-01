#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
	int n, np, my_rank, len, retv = 0;
	char name[MPI_MAX_PROCESSOR_NAME];
	
	MPI_Init(&argc, &argv);
	MPI_Get_processor_name(name, &len);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	if (my_rank == 0) {
		if (argc < 2) {
			printf("Usage: %s size\n", argv[0]);
			retv = -1;
		}
		else { n = atoi(argv[1]); }
	}
	
	// Broadcasts a value from process 0 to all processes in MPI_COMM_WORLD
	MPI_Bcast(&retv, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	if(retv < 0) {
		MPI_Finalize();
		return(EXIT_FAILURE);
	}
	printf("< %s >: process %d of %d\n", name, my_rank, np);
	
	// Broadcasts a value from process 0 to all processes in MPI_COMM_WORLD
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	printf("P%d: received N = %d\n", my_rank, n);
	
	int total = 0;
	int acum  = n + my_rank;
	printf("P%d: after updating (N = %d)\n", my_rank, acum);
	
	// Reduces values on all processes
	MPI_Reduce(&acum, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	printf("P%d: Reduce (total = %d)\n", my_rank, total);
	
	total = 0;
	// Combines values from all processes and distributes the result back to each one.
	MPI_Allreduce(&acum, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	printf("P%d: All_Reduce (total = %d)\n", my_rank, total);
	
	MPI_Finalize();
	return(EXIT_SUCCESS);
}
