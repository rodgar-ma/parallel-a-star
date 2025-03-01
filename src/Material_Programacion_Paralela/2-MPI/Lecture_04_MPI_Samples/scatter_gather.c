#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void init_array(int n, double *v)
{
	srand(time(NULL) + getpid());
	for(int i = 0; i < n; i++) {
		v[i] = (double) rand() / RAND_MAX + 1;
	}
}

void print_array(int n, double *v)
{
	for(int i = 0; i < n; i++) { printf("%g ", v[i]); }
	printf("\n");
}

int main(int argc, char *argv[])
{
	double *v, *r;
	double *loc_v;
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
				v = (double *) calloc(n,  sizeof(double));
				r = (double *) calloc(np, sizeof(double));
				init_array(n, v);
				
				#ifdef DEBUG
					print_array(n, v);
					printf("\n");
				#endif
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
	
	int my_size = n/np;
	loc_v = (double *) malloc(sizeof(double) * my_size);
	
	// Distributes the data from process 0 to all processes
	MPI_Scatter(v, my_size, MPI_DOUBLE, loc_v, my_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	#ifdef DEBUG
		print_array(my_size, loc_v);
		printf("\n");
	#endif
	
	double acum = .0;
	for(int i = 0; i < my_size; i++) {
		acum += loc_v[i];
	}
	printf("P%d: acum = %g\n", my_rank, acum);
	
	// Gathers values from processes in MPI_COMM_WORLD
	MPI_Gather(&acum, 1, MPI_DOUBLE, r, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	if(my_rank == 0) {
		#ifdef DEBUG
			printf("Result: ");
			print_array(np, r);
			printf("\n");
		#endif
		free(r); free(v);
	}
	free(loc_v);
	
	MPI_Finalize();
	return(EXIT_SUCCESS);
}
