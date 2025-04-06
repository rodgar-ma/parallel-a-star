#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct {
	double value;
	int    index;
} PStruct;

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
	double *v, *loc_v;
	int n, np, my_rank, my_size, len, retv = 0;
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
				v = (double *) calloc(n, sizeof(double));
				init_array(n, v);
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
	
	my_size = n/np;
	loc_v = (double *) malloc(sizeof(double) * my_size);
	
	// Distributes the data from process 0 to all processes
	MPI_Scatter(v, my_size, MPI_DOUBLE, loc_v, my_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	#ifdef DEBUG
		print_array(my_size, loc_v);
		printf("\n");
	#endif
	
	
	
	PStruct *in = (PStruct *) malloc(sizeof(PStruct) * my_size);
	for(int i = 0; i < my_size; i++) 
	{
		in[i].value = loc_v[i];
		in[i].index = my_rank;
	}
	
	
	
	PStruct *out;
	if(my_rank == 0) 
	{
		out = (PStruct *) malloc(sizeof(PStruct) * my_size);
	}
	
	
	
	// Reduces values by computing the global minimum for each position and the process rank
	MPI_Reduce(in, out, my_size, MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
	
	
	if(my_rank == 0) 
	{
		#ifdef DEBUG
			printf("Result:\n");
			for(int i = 0; i < my_size; i++) 
			{
				printf("(%g, %d) ", out[i].value, out[i].index);
			}
			printf("\n\n");
		#endif
		free(v); free(out);
	}
	free(loc_v); free(in);
	
	MPI_Finalize();
	return(EXIT_SUCCESS);
}
