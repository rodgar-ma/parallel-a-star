#include <stdlib.h>
#include <cuda.h>
#include "astar.h"
#include "heap.h"
#include "list.h"
#include "cuda_utils.h"

#define THREADS_PER_BLOCK 1024
#define BLOCKS 16
#define RESULT_LEN (1024 * 1024)

__device__ AStarSource *source_gpu;

path *find_path(AStarSource *source, int start_id, int target_id, double *time) {
    int *s_gpu;
    int *t_gpu;
    get_neighbors get_neighbors;
    heuristic heuristic;

    HANDLE_RESULT(cudaMalloc(&s_gpu, sizeof(int)));
    HANDLE_RESULT(cudaMemcpy(s_gpu, &start_id, sizeof(int), cudaMemcpyDefault));
    HANDLE_RESULT(cudaMalloc(&t_gpu, sizeof(int)));
    HANDLE_RESULT(cudaMemcpy(t_gpu, &target_id, sizeof(int), cudaMemcpyDefault));

    HANDLE_RESULT(cudaMemcpyFromSymbol(&get_neighbors, source->get_neighbors, sizeof(get_neighbors)));
    HANDLE_RESULT(cudaMemcpyFromSymbol(&heuristic, source->heuristic, sizeof(heuristic)));

    int k = THREADS_PER_BLOCK * BLOCKS;

    node_t **H;
    heap_t **Q = heaps_init(k);
    int ***expand_buf = expand_buf_create(THREADS_PER_BLOCK * BLOCKS, );

}

int ***expand_buf_create(int bufs, int capacity) {
    int ***bufs_cpu = (int ***)malloc(bufs * sizeof(int**));
    for (int i = 0; i < bufs; i++) {
        bufs_cpu[i] = (int **)malloc(capacity * sizeof(int *));
    }
}