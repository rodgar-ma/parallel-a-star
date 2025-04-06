#include <omp.h>
#include <stdio.h>
#include <unistd.h>

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
    #pragma omp sections
    {
      #pragma omp section
      {
        printf("Soy el thread %d, actuando en solitario dentro de la seccion primera \n", iam);
        sleep(2);
      } // section
      #pragma omp section
      {
        printf("Soy el thread %d, actuando en solitario dentro de la seccion segunda \n", iam);
        sleep(2);
      } // section
      #pragma omp section
      {
        printf("Soy el thread %d, actuando en solitario dentro de la seccion tercera \n", iam);
        sleep(2);
      } // section
      #pragma omp section
      {
        printf("Soy el thread %d, actuando en solitario dentro de la seccion cuarta \n", iam);
        sleep(1);
      } // section
    } // sections

  } // parallel
} // main
