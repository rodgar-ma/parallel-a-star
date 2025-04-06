#if defined (_OPENMP)
	#include <omp.h>
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
	int i;
	int tid, nt, max = 1;
	const long n = 100000000;
	char str[12] = "Hello World";
	
	double acum;
	double h, x, pi, sum;
	double pi21 = 3.14159265358979323846;
	
	#if defined (_OPENMP)
		max = omp_get_max_threads();
	#endif
	
	if(argc < 2) {
		fprintf(stderr, "Usage: %s <omp_threads>\n", argv[0]);
		return(EXIT_FAILURE);
	}
	nt = (max > 1) ? atoi(argv[1]) : max;

	if(nt < 1) {
		fprintf(stderr, "Invalid Value for OMP_THREADS...\n");
		return(EXIT_FAILURE);
	}
	
	printf("\n");
	printf("MAX_OMP_THREADS: %d\n", max);
	printf("OMP_THREADS: %d\n", nt);
	printf("\n");
	
	sum = 0.0;
	h = 1.0 / (double) n;
	
	double ti, tf;
	ti = omp_get_wtime();
	
	#pragma omp parallel for shared(sum) private(i,x,acum) firstprivate(h) num_threads(nt)
	for(i=1; i <= n; i++) {
		x = h*((double) i - 0.5);
		acum = (4.0/(1.0 + x*x));
		#pragma omp atomic
		sum += acum;
	}
	tf = omp_get_wtime();
	
	pi = sum*h;
	printf("Time = %.4lf seconds\n", (tf - ti));
	printf("El valor aproximado de PI es: %1.16lf, con un error de %1.16lf\n", pi, fabs(pi - pi21));
	
	printf("\n");
	return(EXIT_SUCCESS);
}
