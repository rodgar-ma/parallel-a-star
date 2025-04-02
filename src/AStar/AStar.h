#ifndef AStar_h
#define AStar_h

typedef struct __NeighborsList *NeighborsList;

typedef struct {
    int (*Heuristic)(void *fromNode, void *toNode);
    void (*GetNeighbors)(NeighborsList *neighbors, void *node);
    int (*CompareFunc)(void *node1, void *node2);
} AStarSource;

void AddNeighbor(NeighborsList neighbors, void *node, float cost);

#endif