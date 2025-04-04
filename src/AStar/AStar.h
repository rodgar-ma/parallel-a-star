#ifndef AStar_h
#define AStar_h

#include <stdlib.h>

typedef struct __NeighborsList *NeighborsList;
typedef struct __Path *Path;

struct __Path {
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
Path FindPath(AStarSource *source, void *start, void *goal);
void FreePath(Path path);

#endif