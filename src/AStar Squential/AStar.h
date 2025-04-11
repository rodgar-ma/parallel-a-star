#ifndef ASTAR_H
#define ASTAR_H

#include <stdlib.h>

typedef struct __neighbors_list *nieghbors_list;
typedef struct __path *path;

struct __path {
    double cost;
    size_t size;
    void **nodes;
};

typedef struct {
    size_t map_size;
    double (*Heuristic)(void *fromNode, void *toNode);
    void (*GetNeighbors)(NeighborsList neighbors, void *node);
    int (*Equals)(void *a, void *b);
} AStarSource;

void AddNeighbor(NeighborsList neighbors, void *node, double cost);
Path FindPath(AStarSource source, void *start, void *goal);
void FreePath(Path path);

#endif