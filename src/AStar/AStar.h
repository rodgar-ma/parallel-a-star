#ifndef AStar_h
#define AStar_h

typedef void* Node;

typedef struct {
    float (*Heuristic)(Node a, Node b);
    Node* (*NeighborFunc)(Node current, int* count);
    float (*CostFunc)(Node from, Node to);
    int (*CompareFunc)(Node a, Node b);
} AStarSource;


#endif