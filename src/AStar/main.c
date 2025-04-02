#include <stdlib.h>
#include "AStar.h"
#include "MapReader.h"
#include "math.h"

Map MAP;

int ManhattanHeuristic(void *fromNode, void *toNode) {
    return abs(((Node)fromNode)->x - ((Node)toNode)->x) + abs(((Node)fromNode)->y - ((Node)toNode)->y);
}

void GetNeighbors(NeighborsList neighbors, void *node) {
    int x = ((Node)node)->x;
    int y = ((Node)node)->y;
    if (MAP->grid[y-1][x]) AddNeighbor(neighbors, (void*)MAP->grid[y-1][x], 1);
    if (MAP->grid[y+1][x]) AddNeighbor(neighbors, (void*)MAP->grid[y-1][x], 1);
    if (MAP->grid[y][x-1]) AddNeighbor(neighbors, (void*)MAP->grid[y][x-1], 1);
    if (MAP->grid[y][x+1]) AddNeighbor(neighbors, (void*)MAP->grid[y][x+1], 1);
}

int main(int argc, char const *argv[])
{
    if (argc != 2) {
        perror("Uso: <exe> filename");
        return 1;
    }
    const char *filename = argv[1];
    MAP = LoadMap(filename);
    if (!MAP) {
        return 1;
    }



    return 0;
}
