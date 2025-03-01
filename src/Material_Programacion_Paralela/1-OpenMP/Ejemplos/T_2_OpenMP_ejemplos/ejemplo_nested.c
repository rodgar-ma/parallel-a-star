#include <omp.h>
#include <stdio.h>



int main()  
{                                 
	int iam =0, np = 1, i=0,j=0,x;     
	
	x=9999; // lo ponemos en el master
	omp_set_dynamic(0);
        omp_set_num_threads(2);
	
	#pragma omp parallel private(iam, np,i) 
    	{
		#if defined (_OPENMP)
      		  np = omp_get_num_threads(); 
      		  iam = omp_get_thread_num();
                #endif
		
		printf("Hello from thread %d out of %d \n",iam,np);

                omp_set_dynamic(0);
                omp_set_nested(1);
                omp_set_num_threads(3);
                printf("\tNumero de threads actuales = %d \n",omp_get_num_threads());
                
		#pragma omp parallel private(iam, np,i) 
		{
		  #if defined (_OPENMP) 
      		    np = omp_get_num_threads(); 
      		    iam = omp_get_thread_num();
      		  #endif
		
      		  printf("\t\tHello from thread %d out of %d \n",iam,np);
      		  printf("\t\t\tnumero de threads actuales = %d \n",omp_get_num_threads());

                }//parallel
                  
                
        }//parallel
        
}//main

