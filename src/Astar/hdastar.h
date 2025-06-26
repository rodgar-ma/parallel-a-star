#ifndef HDASTAR_H
#define HDASTAR_H

#include <stdlib.h>
#include <omp.h>
#include "astar.h"

#define INIT_QUEUE_CAPACITY 10

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

path *hdastar_search(AStarSource *source, int s_id, int t_id, int k, double *cpu_time_used);

#endif