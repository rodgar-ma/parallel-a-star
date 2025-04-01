#include <stdlib.h>
#include "AStar.h"


struct __NeighborsList {
    int count;
    Neighbor first;
};

struct __Neighbor {
    float cost;
    void *node;
};

NeighborsList CreateNeighborsList(const AStarSource * source) {
    NeighborsList nl = malloc(sizeof(struct __NeighborsList));
    nl->capacity = 0;

}

void AddNeighbor(NeighborsList neighbors, void *node) {
    if (!neighbors) {

    }
}


