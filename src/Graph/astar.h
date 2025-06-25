#ifndef ASTAR_H
#define ASTAR_H

#include <stdlib.h>
#include "graph.h"

#define INIT_NEIGHBORS_LIST_CAPACITY 10

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
    Graph *graph;
    void (*get_neighbors)(neighbors_list *neighbors, int n_id);
    float (*heuristic)(int n1_id, int n2_id);
} AStarSource;

void add_neighbor(neighbors_list *neighbors, int n_id, float cost);

path *astar_search(AStarSource *source, int s_id, int t_id, double *cpu_time_used);

void path_destroy(path *path);

#endif