#include <omp.h>
#include <stdio.h>
#include <unistd.h>

int main()  
{                                 
	int iam,np,i,j;     
	
	#pragma omp parallel private(iam, np,i)
    	{
		#if defined (_OPENMP) 
      		  np = omp_get_num_threads(); 
      		  iam = omp_get_thread_num();
                #endif
		//printf("Hello from thread %d out of %d \n",iam,np);

		#pragma omp critical
		{
                  printf("Soy el thread %d, al inicio de la seccion critica \n",iam);
                  sleep(1);
                  printf("\t\tSoy el thread %d, al final de la seccion critica \n",iam);
                }
                
        }//parallel
}//main

