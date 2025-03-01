#include <omp.h>
#include <stdio.h>
#include <unistd.h>
//ejemplo_atomic.c

int main()  
{                                 
	int iam,np,i,j;     
	int count=0;
	#pragma omp parallel private(iam, np,i)
    	{
		#if defined (_OPENMP) 
      		  np = omp_get_num_threads(); 
      		  iam = omp_get_thread_num();
                #endif
		//printf("Hello from thread %d out of %d \n",iam,np);


                  #pragma omp atomic
		      count++;
   }// parallel
   
   printf("Number of threads: %d\n", count);

}//main

