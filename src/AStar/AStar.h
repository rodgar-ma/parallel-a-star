#ifndef AStar_h
#define AStar_h

#include <stdlib.h>


typedef struct __NeighborRecord *NeighborsList;
typedef struct __NodeRecord *Path;

typedef struct {
    size_t nodeSize;
    NeighborsList (*GetNeighbors)(void *node);
    float (*Heuristic)(void *fromNode, void *toNode);
} AStarSource;

void AddNeighbor(NeighborsList neighbors, void *node, float edgeCost);

Path FindPath(const AStarSource *source, void *start, void *goal);

#endif