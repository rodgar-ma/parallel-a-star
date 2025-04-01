#ifndef AStar_h
#define AStar_h

typedef struct __NeighborsList *NeighborsList;

typedef struct {
    float (*Heuristic)(void *a, void *b);
    void (*GetNeighbors)(NeighborsList neighbors, void *node);
    int (*NodeComparator)(void *a, void *b);
} AStarSource;

void AddNeighbor(NeighborsList neighbors, void *node, float cost);

#endif