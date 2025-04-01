#ifndef AStar_h
#define AStar_h

#include <stdlib.h>


typedef struct __NeighborsList *NeighborsList;
typedef struct __NodeRecord *Path;

typedef struct {
    void (*GetNeighbors)(void *node);
    float (*Heuristic)(void *fromNode, void *toNode);
} AStarSource;

void AddNeighbor(NeighborsList neighbors, void *node, float edgeCost);

Node CreateNode(void *node);

Path FindPath(const AStarSource *source, void *start, void *goal);

#endif