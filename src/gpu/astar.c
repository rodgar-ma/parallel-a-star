#include <math.h>
#include "astar.h"

int d_height;
int d_width;
int d_targetX;
int d_targetY;
u_int32_t d_targetID;
u_int32_t d_modules[10];

void initializeCUDAConstantMemory(int height, int width, int targetX, int targetY, int targetID, const int modules[10]) {
    d_height = height;
    d_width = width;
    d_targetX = targetX;
    d_targetY = targetY;
    d_targetID = targetID;
}

void idToXY(u_int32_t id, int *x, int *y) {
    *x = id / d_width;
    *y = id % d_width;
}

float heuristic(int x, int y) {
    return (float)(sqrt(pow(abs(d_targetX - x), 2) + pow(abs(d_targetY - y), 2)));
}

float heuristic(u_int32_t id) {
    int x, y;
    idToXY(id, &x, &y);
    return heuristic(x, y);
}

void initialize(node_t g_nodes[], u_int32_t g_hash[], heap_t g_openList[], int g_heapSize, int startX, int startY) {
    node_t node;
    node.id = startID;
    node.fCost = heuristic(startX);
    node.gCost = 0;
    node.parent = INT32_MAX;
    
}