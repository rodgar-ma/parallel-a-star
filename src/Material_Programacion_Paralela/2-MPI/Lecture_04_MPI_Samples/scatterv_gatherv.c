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
	double *v, *loc_v;
	int *disp, *count;
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
			v = (double *) calloc(n, sizeof(double));
			init_array(n, v);
			
			#ifdef DEBUG
				print_array(n, v);
				printf("\n");
			#endif
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
	
	disp   = (int *) calloc(np, sizeof(int));	// displacement of each subarray (per each process) from the beginning of the global array
	count = (int *) calloc(np, sizeof(int));	// quantity of data of each subarray (per each process)

	// A sample:
	// n=10
	// np=2
	//
	// index of v	:	0	1	2	3	4	5	6	7	8	9
	// to process	:	0	0	0	1	1	1	2	2	3	3
	//
	//
	// count	:	3	3	2	2
	// disp		:	0	3	6	8
	//
	
	int offset = 0;
	int mod = (n % np);
	for(int i = 0; i < np; i++) {
		count[i] = (i == np-1) ? (n - offset) : n/np;
		if(mod > 0) {
			count[i]++;
			mod--;
		}
		disp[i] = offset;
		offset += count[i];
	}
	
	int my_size = count[my_rank];
	loc_v = (double *) malloc(sizeof(double) * my_size);
	
	// Scatters a buffer in parts to all processes in MPI_COMM_WORLD
	MPI_Scatterv(v, count, disp, MPI_DOUBLE, loc_v, my_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	#ifdef DEBUG
		print_array(my_size, loc_v);
		printf("\n");
	#endif
	
	for(int i = 0; i < my_size; i++) {
		loc_v[i] += (i + my_rank);
	}
	
	// Gathers varying amounts of data from all processes to process 0
	MPI_Gatherv(loc_v, my_size, MPI_DOUBLE, v, count, disp, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	if(my_rank == 0) {
		#ifdef DEBUG
			printf("Result: ");
			print_array(n,  v);
		#endif
		free(v);
	}
	free(loc_v);
	free(count); free(disp);
	
	MPI_Finalize();
	return(EXIT_SUCCESS);
}
