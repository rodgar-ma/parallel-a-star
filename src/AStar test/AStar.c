#include <stdlib.h>
#include "AStar.h"

typedef struct __Neighbor *Neighbor;

struct __NeighborsList {
    int count;
    Neighbor first;
};

struct __Neighbor {
    float cost;
    void *node;
    Neighbor next;
};

Neighbor CreateNeighbor(void *node, float cost) {
    Neighbor nb = malloc(sizeof(struct __Neighbor));
    nb->cost = cost;
    nb->node = node;
}

void AddNeighbor(NeighborsList neighbors, void *node, float cost) {
    Neighbor nb = CreateNeighbor(node, cost);

}


