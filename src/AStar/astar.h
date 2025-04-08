#ifndef ASTAR_H
#define ASTAR_H

#include <stdlib.h>

typedef struct __path path;

typedef struct __node {
    void *node;
    node *parent;
    double gCost;
    double fCost;
} node;

typedef struct __neighbors_list {
    size_t capacity;
    size_t count;
    double *costs;
    void **elements;
} neighbors_list;

typedef struct {
    void (*get_neighbors)(void *node, neighbors_list *neighbors);
    double (*heuristic)(void *node1, void *node2);
} AStarSource;

path *find_path(AStarSource source, void *start, void *target, int k);
void add_neighbor(neighbors_list neighbors, void *node, double cost);
double path_get_cost(path *path);
int path_get_count(path *path);


#endif