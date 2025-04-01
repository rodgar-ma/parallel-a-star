#ifndef AStar_h
#define AStar_h

typedef struct __NeighborsList *NeighborsList;

typedef struct {
    void (*GetNeighbors)(NeighborsList neighbors, void *node);
    float (*Heuristic)(void *node1, void *node2);
} AStarSource;

#endif