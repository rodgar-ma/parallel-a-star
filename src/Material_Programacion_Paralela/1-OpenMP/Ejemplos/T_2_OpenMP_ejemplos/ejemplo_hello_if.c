#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(argc,argv)  
int argc;
char *argv[];
{                                 
  int iam=0, np=1;     

  int decision;
  
      decision=atoi(argv[1]);
  
      printf("decision=%d \n",decision);
  
      #pragma omp parallel private(iam, np) if(decision)
      {
		#if defined (_OPENMP) 
      		np = omp_get_num_threads(); 
      		iam = omp_get_thread_num();
		#endif
                
                printf("Hello from thread %d out of %d \n",iam,np);	
      }
}

