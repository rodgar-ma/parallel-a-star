////////////////////////////////////////////////////////////////////////////////
// Dummy kernel
////////////////////////////////////////////////////////////////////////////////

__constant__ int constante_d[CM_SIZE];
//__device__ int basura[16];

__global__ void foo(int *gid_d)
{
     extern __shared__ int shared_mem[];

    int blockSize = blockDim.x * blockDim.y;

    // global thread ID in thread block
    int tidb = (threadIdx.y * blockDim.x + threadIdx.x);

    // global thread ID in grid
    int tidg = (blockIdx.y * gridDim.x * blockSize + blockIdx.x * blockSize + tidb);

    shared_mem[tidb] = gid_d[tidg];
    
    __syncthreads();

    shared_mem[tidb] = (tidg+constante_d[tidg%CM_SIZE]);

    __syncthreads();

    gid_d[tidg] = shared_mem[tidb];
}

