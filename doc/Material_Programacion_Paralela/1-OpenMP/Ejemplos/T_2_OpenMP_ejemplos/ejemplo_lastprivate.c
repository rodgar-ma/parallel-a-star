#include <omp.h>
#include <stdio.h>

//ejemplo_last_private.c

int main()  
{                                 
        int x=9999;
	int iam,np,i,j;     
	
	#pragma omp parallel private(iam, np,i) 
    	{
		#if defined (_OPENMP) 
      		  np = omp_get_num_threads(); 
      		  iam = omp_get_thread_num();
                #endif
		//printf("Hello from thread %d out of %d \n",iam,np);

         
                  printf("Soy el thread %d, antes del for, con x=%d \n",iam,x);  
                  
                  #pragma omp for lastprivate(x) // schedule(dynamic)
                  for(i=0;i<11;i++)
                  {
                      printf("\tSoy el thread %d, antes de actualizar en for, i=%d x=%d \n",iam,i,x);
                      x=iam*i;
                      printf("\tSoy el thread %d, despues de actualizar en for, i=%d x=%d \n",iam,i,x);
                      
                  }
                  
                  printf("\t\tSoy el thread %d, despues del for, con x=%d \n",iam,x); 
                  //todos se quedan con el valor que ponga el thread que haga la ultima iteracion, es decir, i=10
                
                
        }//parallel
        
        printf("\n Despues de pragma parallel x=%d \n\n",x);
}//main

