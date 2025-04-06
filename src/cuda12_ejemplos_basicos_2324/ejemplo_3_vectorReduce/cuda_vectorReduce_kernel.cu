////////////////////////////////////////////////////////////////////////////////
// vectorReduce kernel
////////////////////////////////////////////////////////////////////////////////

__global__ void vectorReduce(float *vector_d, float *reduce_d, int n)
{
    extern __shared__ int sdata[];

    // global thread ID in thread block
    unsigned int tidb = threadIdx.x;
    
    // global thread ID in grid
    unsigned int tidg = blockIdx.x * blockDim.x + threadIdx.x;


	//printf("blockIdx.x=%d threadIdx.x=%d \n",blockIdx.x,threadIdx.x);

    // load shared memory
    sdata[tidb] = (tidg < n) ? vector_d[tidg]: 0;

    __syncthreads();
     
    // perform reduction in shared memory
    for(unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tidb < s) {
            sdata[tidb] += sdata[tidb + s];
        }
        __syncthreads();
    }
                        
    // write result for this block to global memory
    if (tidb == 0) {
        reduce_d[blockIdx.x] = sdata[0];
    }
}
