
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int THREADS;







///////////////////////////////////////////////////////////////

// para matrices cuadradas nxn con lead dimension n tambien
//
// bucle j paralelo
//////////////////////////////////////////////////////////////

void mm_j(double *a, double *b, double *c, int n)
{
  int i,j,iam,nprocs,k;
  double s;
  int	tb;
  #pragma omp parallel private(iam,nprocs,i) 
  {
    nprocs=omp_get_num_threads();
    iam=omp_get_thread_num();
    tb=1;//n/nprocs;
    #pragma omp master
      THREADS=nprocs;
    
    
    for (i = 0; i < n; i++) 
    {
      //printf("iteracion i:%d, thread %d \n",i,iam);
      
      #pragma omp for private(s,j) schedule(static,tb)
      for (j=0;j<n;j++)
      {
        //printf("\t\titeracion i:%d j:%d, thread %d \n",i,j,iam);
        s=0.;
        for(k=0;k<n;k++)
          s+=a[i*n+k]*b[k*n+j];
        c[i*n+j]=s;
      }//for j
    }//for i
  }//pragma parallel
}


///////////////////////////////////////////////////////////////

// para matrices cuadradas nxn con lead dimension n tambien
//
// bucle i paralelo
//////////////////////////////////////////////////////////////

void mm_i(double *a, double *b, double *c, int n)
{
  int i,j,iam,nprocs,k;
  double s;
  int	tb;
  #pragma omp parallel private(iam,nprocs) 
  {
    nprocs=omp_get_num_threads();
    iam=omp_get_thread_num();
    tb=1;//n/nprocs;
    #pragma omp master
      THREADS=nprocs;
    
    
    #pragma omp for private(i,s,j) schedule(static,tb)
    for (i = 0; i < n; i++) 
    {
      //printf("iteracion %d, thread %d \n",i,iam);
      for (j=0;j<n;j++)
      {
        s=0.;
        for(k=0;k<n;k++)
          s+=a[i*n+k]*b[k*n+j];
        c[i*n+j]=s;
      }//for j
    }//for i
  }//pragma parallel
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
  double *a,*b,*c;
  
  if (argc<4)
  {
    printf("\n\n USO %s tamanyo_inicial tamanyo_final incremento_tamanyo \n\n",argv[0]);
    return -1;
  }
  
  iN=atoi(argv[1]);
  fN=atoi(argv[2]);
  incN=atoi(argv[3]);
  for(t=iN;t<=fN;t+=incN) {
    a = (double *) malloc(sizeof(double)*t*t);
    b = (double *) malloc(sizeof(double)*t*t);
    c = (double *) malloc(sizeof(double)*t*t);
    initializealea(a,t*t);
    initializealea(b,t*t);
    
    #ifdef DEBUG
      escribir(a,t,t,t);
      escribir(b,t,t,t);
    #endif
    start=omp_get_wtime();
    mm_i(a,b,c,t);
    fin=omp_get_wtime();
    
    tiempo=fin-start;
    if(tiempo==0.) {
      printf("No hay suficiente precision\n");
    }
    else {
      Mflops=((2.*t*t*t)/tiempo)/1000000.;
      printf("  MM_i \n threads %d, tamano %d\n    segundos: %.6lf, Mflops: %.6lf, Mflops por thread: %.6lf\n",THREADS,t,tiempo,Mflops,Mflops/THREADS);
      printf("  Precision omp_get_wtick: numero de segundos entre sucesivos ticks de reloj usados por wtime=%lf \n",omp_get_wtick());
    }
    #ifdef DEBUG
      escribir(c,t,t,t);
    #endif
    free(a);
    free(b);
    free(c);
  }//for
  
  
  
  
    for(t=iN;t<=fN;t+=incN) {
    a = (double *) malloc(sizeof(double)*t*t);
    b = (double *) malloc(sizeof(double)*t*t);
    c = (double *) malloc(sizeof(double)*t*t);
    initializealea(a,t*t);
    initializealea(b,t*t);
    
    #ifdef DEBUG
      escribir(a,t,t,t);
      escribir(b,t,t,t);
    #endif
    start=omp_get_wtime();
    mm_j(a,b,c,t);
    fin=omp_get_wtime();
    
    tiempo=fin-start;
    if(tiempo==0.) {
      printf("No hay suficiente precision\n");
    }
    else {
      Mflops=((2.*t*t*t)/tiempo)/1000000.;
      printf("  MM_j \n threads %d, tamano %d\n    segundos: %.6lf, Mflops: %.6lf, Mflops por thread: %.6lf\n",THREADS,t,tiempo,Mflops,Mflops/THREADS);
      printf("  Precision omp_get_wtick: numero de segundos entre sucesivos ticks de reloj usados por wtime=%lf \n",omp_get_wtick());
    }
    #ifdef DEBUG
      escribir(c,t,t,t);
    #endif
    free(a);
    free(b);
    free(c);
  }//for
  
  
}
