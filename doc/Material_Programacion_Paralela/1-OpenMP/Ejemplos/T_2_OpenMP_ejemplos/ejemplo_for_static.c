#include <stdio.h>
#include <omp.h>

// ejemplo_for_static

int main()  
{                                 
	int iam,np,i=2;     
	#pragma omp parallel private(iam, np,i)
    	{
		#if defined (_OPENMP) 
      		  np = omp_get_num_threads(); 
      		  iam = omp_get_thread_num();
                #endif
		//printf("Hello from thread %d out of %d \n",iam,np);

		#pragma omp for schedule(static,2) 
		for(i=0;i<(np*4);i++)
		{
		  printf("Thread %d, contador %d \n",iam,i);
                }
    	
    }
}

