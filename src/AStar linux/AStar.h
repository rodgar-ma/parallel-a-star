#ifndef AStar_h
#define AStar_h

typedef struct __NeighborsList *NeighborsList;
typedef struct __Path *Path;

struct __Path {
    int cost;
    size_t size;
    void **nodes;
};

typedef struct {
    int map_size;
    int (*Heuristic)(void *fromNode, void *toNode);
    void (*GetNeighbors)(NeighborsList neighbors, void *node);
} AStarSource;

void AddNeighbor(NeighborsList neighbors, void *node, int cost);

Path FindPath(AStarSource *source, void *start, void *goal);

#endif