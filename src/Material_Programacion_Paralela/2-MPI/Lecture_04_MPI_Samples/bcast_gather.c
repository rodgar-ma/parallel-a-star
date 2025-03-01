#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void print_array(int n, int *val)
{
	for(int i = 0; i < n; i++) { printf("%d ", val[i]); }
	printf("\n");
}

int main(int argc, char *argv[])
{
	int *values = NULL;
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
		else {
			n = atoi(argv[1]);
			if((n % np) != 0) {
				printf("N must be divisible by NP...\n");
				retv = -2;
			}
			else {
				values = (int *) calloc(np, sizeof(int));
			}
		}
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
	
	int acum = n + my_rank;
	printf("P%d: after updating (N = %d)\n", my_rank, acum);
	
	// Process 0 gathers values from processes in MPI_COMM_WORLD
	MPI_Gather(&acum, 1, MPI_INT, values, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	if(my_rank == 0) {
		#ifdef DEBUG
			printf("Result: ");
			print_array(np, values);
		#endif
		free(values);
	}
	
	MPI_Finalize();
	return(EXIT_SUCCESS);
}
