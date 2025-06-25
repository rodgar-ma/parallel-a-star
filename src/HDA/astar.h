#ifndef ASTAR_H
#define ASTAR_H

#include <stdlib.h>
#include <omp.h>

#define INIT_NEIGHBORS_LIST_CAPACITY 10
#define INIT_QUEUE_CAPACITY 10

typedef struct node_t {
    int id;
    float gCost;
    float fCost;
    int parent;
    unsigned int is_open:1;
    int open_index;
} node_t;

typedef struct neighbors_list {
    int capacity;
    int count;
    int *nodeIds;
    float *costs;
} neighbors_list;

typedef struct path {
    int count;
    int *nodeIds;
    float cost;
} path;

typedef struct {
    int node_id;
    float gCost;
    int parent_id;
} queue_elem_t;

typedef struct {
    int size;
    int capacity;
    queue_elem_t *elems;
    omp_lock_t lock;
} queue_t;

typedef struct {
    int size;
    int capacity;
    queue_elem_t *elems;
} buffer_t;

typedef struct {
    int max_size;
    void (*get_neighbors)(neighbors_list *neighbors, int n_id);
    float (*heuristic)(int n1_id, int n2_id);
} AStarSource;

void add_neighbor(neighbors_list *neighbors, int n_id, float cost);

path *astar_search(AStarSource *source, int s_id, int t_id, int k, double *cpu_time_used);

void path_destroy(path *path);

#endif