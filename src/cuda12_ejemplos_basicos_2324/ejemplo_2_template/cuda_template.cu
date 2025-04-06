/* -------------------------------------------------------------------------- */
/* Project: I Curso de Computación Científica en Clusters                     */
/* Author:  Juan Fernández Peinador                                           */
/* Date:    Marzo de 2010                                                     */
/*										*/
/* Modificado en Julio 2014 para cuda 5.0					*/
/* Modificado en Enero 2020: introducción parametros dimensiones bloque y de grid*/

/* Modificado en Febrero 2021: funcion cudaDeviceReset actualizado		*/										 
/* -------------------------------------------------------------------------- */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

// includes, project
#include <cuda.h>
#include <cuda_runtime.h>

 #include <helper_functions.h>
 #include <helper_cuda.h>
 #include <timer.h>
 #include <helper_string.h>


#define SHM_SIZE (16 * 1024)
#define CM_SIZE  (8)

// includes, kernels
#include "cuda_template_kernel.cu"

const int constante_h[CM_SIZE] = {1, 2, 3, 4, 5, 6, 7, 8};

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    int *gid_h; // host data
    int *gid_d; // device data
    int shared_mem_size;
    long int nPos;
    size_t nBytes;
    

    ///*events*/    
    float processing_time;
    cudaEvent_t start_event, stop_event;    
    
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;


    // default values
    int tam_grid_x  = 2;
    int tam_grid_y  = 2;
    int tam_block_x = 2;
    int tam_block_y = 2;
    //int tam_block_z = 1;

    // General initialization call to pick the best CUDA Device
    // If the command-line has a device number specified, use it
    // Otherwise pick the device with highest Gflops/s
    //cutilChooseCudaDevice(argc, argv);

    // process command line arguments
    tam_grid_x = getCmdLineArgumentInt(argc, (const char **) argv,  (const char *)"gsx")?:tam_grid_x;
    tam_grid_y = getCmdLineArgumentInt(argc, (const char **) argv, (const char *) "gsy")?:tam_grid_y;
    tam_block_x = getCmdLineArgumentInt(argc, (const char **) argv, (const char *) "bsx")?:tam_block_x;
    tam_block_y = getCmdLineArgumentInt(argc, (const char **) argv, (const char *) "bsy")?:tam_block_y;


    printf("\n\nDimension malla: tam_grid_x=%d tam_grid_y=%d \n", tam_grid_x,tam_grid_y);  
    printf("Dimension bloque: tam_block_x=%d tam_grid_y=%d\n\n",tam_block_x,tam_block_y);


    nPos = tam_grid_x * tam_grid_y * tam_block_x * tam_block_y;
    nBytes = nPos * sizeof(int);

    // allocate host memory
    gid_h = (int *) malloc(nBytes);
    bzero(gid_h, nBytes);

    // allocate device memory
    checkCudaErrors(cudaMalloc((void **) &gid_d, nBytes));
   
    // copy data from host memory to device memory
    checkCudaErrors(cudaMemcpy(gid_d, gid_h, nBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset((void *) gid_d, 0, nBytes));

    // initialize constant memory
    checkCudaErrors(cudaMemcpyToSymbol(constante_d,constante_h, CM_SIZE*sizeof(int)));


   //create events
   checkCudaErrors(cudaEventCreate(&start_event));
   checkCudaErrors(cudaEventCreate(&stop_event));    
    

   ///*using event*/    
   cudaEventRecord(start_event, 0);     // record in stream-0, to ensure that all previous CUDA calls have completed


    // setup execution parameters
    dim3 grid(tam_grid_x, tam_grid_y);
    dim3 block(tam_block_x, tam_block_y);

    // execute the kernel
    shared_mem_size = block.x * block.y * sizeof(int);
    assert(shared_mem_size <= SHM_SIZE);
    printf("Running configuration: \t %ld threads\n\t\t\t grid of %d x %d\n"
           "\t\t\t blocks of %d x %d threads (%d threads with %d bytes of shared memory per block)\n", 
           nPos,
           tam_grid_x, tam_grid_y, tam_block_x, tam_block_y,
           (tam_block_x * tam_block_y), shared_mem_size);
    
    foo<<<grid, block, shared_mem_size>>>(gid_d);

    // wait for thread completion
    //cudaThreadSynchronize();
    cudaDeviceSynchronize();


    // get results back from device memory
    checkCudaErrors(cudaMemcpy(gid_h, gid_d, nBytes, cudaMemcpyDeviceToHost));

    
    
    ///*using event*/    
    cudaEventRecord(stop_event, 0);    
    cudaEventSynchronize(stop_event);   // block until the event is actually recorded    
    checkCudaErrors(cudaEventElapsedTime(&processing_time, start_event, stop_event));    
    printf("Processing time: %f (ms)", processing_time);
    

    // check results
    for(int i = 0; i < nPos; i++)
        assert(gid_h[i] == (i+constante_h[i%CM_SIZE]));


    // destroy events
    cudaEventDestroy(start_event);    
    cudaEventDestroy(stop_event);    
    
    // free memory
    free(gid_h);
    checkCudaErrors(cudaFree((void *) gid_d));

    printf("\nTest PASSED\n");





    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }




}

