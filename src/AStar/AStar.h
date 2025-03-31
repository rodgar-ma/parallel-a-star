#ifndef AStar_h
#define AStar_h

#include <stdlib.h>


typedef struct NeighborsList *NeighborsList;
typedef struct __NodeRecord *Path;

typedef struct {
    void    (*GetNeighbors)(NeighborsList neighbors, void *node, void *context);
    float   (*Heuristic)(void *fromNode, void *toNode, void *context);
} AStarSource;

void AddNeighbor(NeighborsList neighbors, void *node, float edgeCost);

Path FindPath(const AStarSource *source, void *start, void *goal);

#endif