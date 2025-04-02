#include <stdlib.h>
#include <stdio.h>
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
    if (MAP->grid[y+1][x]) AddNeighbor(neighbors, (void*)MAP->grid[y+1][x], 1);
    if (MAP->grid[y][x-1]) AddNeighbor(neighbors, (void*)MAP->grid[y][x-1], 1);
    if (MAP->grid[y][x+1]) AddNeighbor(neighbors, (void*)MAP->grid[y][x+1], 1);
}

void PrintPath(Path path) {
    printf("Path found!\n");
    printf("Number of nodes = %d\n", path->size);
    printf("Total cost = %d\n", path->cost);
    for (int i = 0; i < path->size; i++) {
        printf("[%d,%d]\n", ((Node)path->nodes[i])->x, ((Node)path->nodes[i])->y);
    }
    printf("\n");
}

int main(int argc, char const *argv[])
{
    if (argc != 6) {
        perror("Uso: <exe> filename start_x start_y goal_x goal_y");
        return 1;
    }
    const char *filename = argv[1];
    MAP = LoadMap(filename);
    if (!MAP) {
        return 1;
    }

    int start_x = atoi(argv[2]);
    int start_y = atoi(argv[3]);
    int goal_x = atoi(argv[4]);
    int goal_y = atoi(argv[5]);

    AStarSource *source = malloc(sizeof(AStarSource));
    source->Heuristic = &ManhattanHeuristic;
    source->GetNeighbors = &GetNeighbors;

    Node start = GetNodeAtPos(MAP, start_x, start_y);
    Node goal = GetNodeAtPos(MAP, goal_x, goal_y);

    Path path = FindPath(source, start, goal);

    PrintPath(path);

    FreeMap(MAP);
    return 0;
}
