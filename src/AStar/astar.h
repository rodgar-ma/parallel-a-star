#ifndef ASTAR_H
#define ASTAR_H

#include <stdlib.h>

typedef struct __node node;
typedef struct __path path;
typedef struct __neighbors_list neighbors_list;

struct __node {
    void *node;
    node *parent;
    double gCost;
    double fCost;
};

struct __path {
    size_t count;
    double cost;
    void **nodes;
};

struct __neighbors_list {
    size_t capacity;
    size_t count;
    double *costs;
    void **elements;
};

typedef struct {
    void (*get_neighbors)(void *node, neighbors_list *neighbors);
    double (*heuristic)(void *node1, void *node2);
} AStarSource;

path *find_path(AStarSource source, void *start, void *target, int k);
void add_neighbor(neighbors_list *neighbors, void *node, double cost);
void path_destroy(path *path);


#endif