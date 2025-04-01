#ifndef AStar_h
#define AStar_h

typedef struct {
    void (*GetNeighbors)(void *node);
    float (*Heuristic)(void *node1, void *node2);
} AStarSource;

#endif