#include <omp.h>
#include <stdio.h>
#include <unistd.h>

// ejemplo_single.c

int main()
{
  int iam, np, i;
  #pragma omp parallel private(iam, np, i)
  {
    #if defined(_OPENMP)
    np = omp_get_num_threads();
    iam = omp_get_thread_num();
    #endif
    // printf("Hello from thread %d out of %d \n",iam,np);

    #pragma omp single
    {
      printf("Soy el thread %d, actuando en solitario dentro del primer bloque\n", iam);
      sleep(1);
    }
    #pragma omp single
    {
      printf("Soy el thread %d, actuando en solitario dentro del segundo bloque \n", iam);
      sleep(1);
    }
    #pragma omp single
    {
      printf("Soy el thread %d, actuando en solitario dentro del tercer bloque \n", iam);
      sleep(1);
    }

    printf("Soy el thread %d, fuera de los singles\n", iam);

  } // parallel
} // main
