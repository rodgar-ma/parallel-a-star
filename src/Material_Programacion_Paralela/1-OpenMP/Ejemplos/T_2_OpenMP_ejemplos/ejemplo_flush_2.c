#include <omp.h>
#include <stdio.h>
#include <unistd.h>
// ejemplo_flush_2.c


int main()
{
        int data;
        int flag=0;
        int iam;

        #pragma omp parallel sections num_threads(2)  private(iam)
        {
                #pragma omp section
                {
                        iam=omp_get_thread_num();
                        printf("Thread %d, esperando dato teclaado por usuario: \n",iam);
                        scanf("%d",&data);

                        #pragma omp flush(data)                        
	      	        flag = 1;
                        #pragma omp flush(flag)
                        printf("Thread %d, ha puesto en común el dato tecleado por el usuario: %d\n",iam,data);
                }


		#pragma omp section
                {
                        iam=omp_get_thread_num();
                        printf("ANTES: Thread %d, dato a procesar = %d. Esperando dato desde el otro thread..\n",iam,data);

                        while (!flag)
                        {
                               #pragma omp flush(flag)
                        }       
                        #pragma omp flush(data)
                        printf("DESPUES: Thread %d: dato a procesar = %d \n", iam,data);

                }
        } //parallel
}


