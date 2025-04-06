#include <omp.h>
#include <stdio.h>

//ejemplo_firstprivate.c

int main()  
{                                 
        int x=9999;
	int iam,np,i,j;     
	
        printf("\n Antes de pragma parallel x=%d \n\n",x);	
	
	#pragma omp parallel firstprivate(iam, np,i,x)
    	{
		#if defined (_OPENMP) 
      		  np = omp_get_num_threads(); 
      		  iam = omp_get_thread_num();
                #endif
		//printf("Hello from thread %d out of %d \n",iam,np);

         
                  printf("Soy el thread %d, antes de actualizar, con x=%d \n",iam,x);  
                  x=iam*1111;
                  printf("\t\tSoy el thread %d, despues de actualizar, con x=%d \n",iam,x);
                
                
        }//parallel
        
        printf("\n Despues de pragma parallel x=%d \n\n",x);
}//main

