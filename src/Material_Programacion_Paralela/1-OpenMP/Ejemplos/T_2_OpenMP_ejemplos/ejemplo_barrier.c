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

                  printf("Soy el thread %d, antes del barrier \n",iam);
                  #pragma omp barrier
                  printf("\t\tSoy el thread %d, despues del barrier \n",iam);
                
                
        }//parallel
}//main

