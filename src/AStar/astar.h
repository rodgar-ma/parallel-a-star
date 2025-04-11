#ifndef ASTAR_H
#define ASTAR_H

#include <stdlib.h>

typedef struct __node node;
typedef struct __neighbors_list neighbors_list;
typedef struct __path path;

struct __node {
    unsigned long id;
    node *parent;
    double gCost;
    double fCost;
};

struct __neighbors_list {
    size_t capacity;
    size_t count;
    double *costs;
    unsigned long **nodes;
};

struct __path {
    size_t count;
    double cost;
    unsigned long **nodes;
};

typedef struct {
    size_t max_size;
    void (*get_neighbors)(unsigned long node, neighbors_list *neighbors);
    double (*heuristic)(unsigned long node1, unsigned long node2);
} AStarSource;

path *find_path_sequential(AStarSource *source, unsigned long *start, unsigned long *goal);
path *find_path_openmp(AStarSource *source, unsigned long *start, unsigned long *goal);
void add_neighbor(neighbors_list *neighbors, unsigned long *node, double cost);
void path_destroy(path *path);


#endif