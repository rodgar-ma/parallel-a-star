#include <omp.h>
#include <stdio.h>
#include <unistd.h>

//ejemplo_master.c

int main()  
{                                 
	int iam,np,i;     
	#pragma omp parallel private(iam, np,i)
    	{
		#if defined (_OPENMP) 
      		  np = omp_get_num_threads(); 
      		  iam = omp_get_thread_num();
                #endif
		//printf("Hello from thread %d out of %d \n",iam,np);

                #pragma omp master
                  {
                    printf("Soy el thread %d, actuando en solitario dentro del primer bloque\n",iam);
                    sleep(1);
                  }
                  #pragma omp master
                  {
                    printf("Soy el thread %d, actuando en solitario dentro ddel segundo bloque \n",iam);
                    sleep(1);
                  }
                   #pragma omp master
                  {
                    printf("Soy el thread %d, actuando en solitario dentro ddel tercer bloque \n",iam);
                    sleep(1);
                  }
                  
                  printf("Soy el thread %d, fuera de los bloques de master\n",iam);
                  
                
        }//parallel
}//main

