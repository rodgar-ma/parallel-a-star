#include "astar.h"
#include "heap.h"

#define THREADS_PER_BLOCK 1024
#define BLOCKS 16
#define RESULT_LEN (1024 * 1024)

path *find_path(AStarSource *source, int start_id, int target_id, double *time) {
    int *s_gpu, t_gpu;
    int k = THREADS_PER_BLOCK * BLOCKS;

    heap_t **Q = heaps_init(k);
    path *result = malloc(RESULT_LEN * sizeof(path));
    int result_len = 0;


}