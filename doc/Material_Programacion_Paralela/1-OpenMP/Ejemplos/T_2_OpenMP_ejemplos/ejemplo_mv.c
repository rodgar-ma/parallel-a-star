// Introducción a la Programación Paralela
// capítulo 4: códigos 4.6 y 4.7
// multiplicación matriz-vector en OpenMP

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int THREADS;







///////////////////////////////////////////////////////////////


void mv(double *m,int fm,int cm,int ldm,double *v,int fv,double *w,int fw) {
  int i,j,iam,nprocs;
  double s;
  #pragma omp parallel private(iam,nprocs) 
  {
    nprocs=omp_get_num_threads();
    iam=omp_get_thread_num();
  
    #pragma omp master
      THREADS=nprocs;
  
    #pragma omp for private(i,s,j) schedule(static)
    for (i = 0; i < fm; i++) {
      #ifdef DEBUG
        printf("thread %d fila %d \n",iam,i);
      #endif
  
      s=0.;
      for(j=0;j<cm;j++)
        s+=m[i*ldm+j]*v[j];
      w[i]=s;
    }
  }
}


///////////////////////////////////////////////////////////////




void initialize(double *m,int t) {
  int i;
  for (i = 0; i < t; i++)
    m[i] = (double)(i);
}

///////////////////////////////////////////////////////////////
void initializealea(double *m, int t) {
  int i;
  for (i = 0; i < t; i++)
    m[i] = (double)rand()/RAND_MAX;
}
///////////////////////////////////////////////////////////////
void escribir(double *m, int fm,int cm,int ldm) {
  int i,j;
  for (i = 0; i < fm; i++) {
    for(j=0;j<cm;j++)
      printf("%.4lf ",m[i*ldm+j]);
    printf("\n");
  }
}




///////////////////////////////////////////////////////////////
int main(int argc,char *argv[]) {
  int t,iN,fN,incN;
  double start,fin,tiempo,Mflops;
  double *a,*v,*w;
  
  if (argc<4)
  {
    printf("\n\n USO %s tamanyo_inicial tamanyo_final incremento_tamanyo \n\n",argv[0]);
    return -1;
  }
  
  iN=atoi(argv[1]);
  fN=atoi(argv[2]);
  incN=atoi(argv[3]);

  for(t=iN;t<=fN;t+=incN) 
  {
  
    a = (double *) malloc(sizeof(double)*t*t);
    v = (double *) malloc(sizeof(double)*t);
    w = (double *) malloc(sizeof(double)*t);

    initializealea(a,t*t);
    initializealea(v,t);
    
    #ifdef DEBUG
      escribir(a,t,t,t);
      escribir(v,1,t,t);
    #endif
    start=omp_get_wtime();
    mv(a,t,t,t,v,t,w,t);
    fin=omp_get_wtime();
    
    tiempo=fin-start;
    if(tiempo==0.) {
      printf("No hay suficiente precision\n");
    }
    else {
      Mflops=((2.*t*t)/tiempo)/1000000.;
      printf("\n  Threads %d, tamano %d\n    segundos: %.6lf, Mflops: %.6lf, Mflops por thread: %.6lf\n",THREADS,t,tiempo,Mflops,Mflops/THREADS);
      //printf("  Precision omp_get_wtick: numero de segundos entre sucesivos ticks de reloj usados por wtime=%lf \n",omp_get_wtick());
    }
  
    #ifdef DEBUG
      escribir(w,1,t,t);
    #endif
  
    free(a);
    free(v);
    free(w);
  }
}
