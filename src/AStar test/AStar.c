#include <stdlib.h>
#include "AStar.h"


struct __NeighborsList {
    const AStarSource *source;
    size_t capacity;
    size_t count;
    float *costs;
    void *nodes;
};

struct Node {
    
}
