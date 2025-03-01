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
	double *v, *buf;
	double *local_v;
	int n, np, my_rank, len, position, retv = 0;
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
	
	int acum = 0;
	int my_size = n/np;
	int n_bytes = sizeof(int) + sizeof(double)*my_size;
	buf = (double *) malloc(n_bytes);
	
	if (my_rank == 0) 
        {
            for(int i = 1; i < np; i++) 
            {
	    	position = 0;
		acum = n + i;
		MPI_Pack(&v[i*my_size], my_size, MPI_DOUBLE, buf, n_bytes, &position, MPI_COMM_WORLD);
		MPI_Pack(&acum, 1, MPI_INT, buf, n_bytes, &position, MPI_COMM_WORLD);
		MPI_Send(buf, n_bytes, MPI_PACKED, i, 10, MPI_COMM_WORLD);
            }
        } 
        else 
        {
                position = 0;
                local_v = (double *) malloc(sizeof(double) * my_size);
                MPI_Recv(buf, n_bytes, MPI_PACKED, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Unpack(buf, n_bytes, &position, local_v, my_size, MPI_DOUBLE, MPI_COMM_WORLD);
                MPI_Unpack(buf, n_bytes, &position, &acum, 1, MPI_INT, MPI_COMM_WORLD);
	}
	
	
	if (my_rank == 0) 
	{ 
	        free(v); 
        }
	else 
	{
		#ifdef DEBUG
			printf("P%d: acum = %d\n", my_rank, acum);
			print_array(my_size, local_v);
		#endif
		free(local_v);
	}
	free(buf);
	
	MPI_Finalize();
	return(EXIT_SUCCESS);
}
