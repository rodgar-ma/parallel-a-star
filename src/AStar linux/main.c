#include <stdlib.h>
#include <stdio.h>
#include "AStar.h"
#include "MapReader.h"
#include "math.h"

Map MAP;

int ManhattanHeuristic(void *fromNode, void *toNode) {
    int distX = abs(((Node)toNode)->x - ((Node)fromNode)->x);
    int distY = abs(((Node)toNode)->y - ((Node)fromNode)->y);
    if (distX > distY) return distX;
    else return distY;
}

void GetNeighbors(NeighborsList neighbors, void *node) {
    int x = ((Node)node)->x;
    int y = ((Node)node)->y;
    printf("GetNeighbors: [%d,%d]\n", x, y);
    for (int j = -1; j < 2; j++) {
        for (int i = -1; i < 2; i++) {
            if (i == 0 && j == 0) continue;
            if (MAP->grid[y+j][x+i]) {
                AddNeighbor(neighbors, (void*)MAP->grid[y+j][x+i], 1);
                printf("\t[%d,%d]\n", ((Node)MAP->grid[y+j][x+i])->x, ((Node)MAP->grid[y+j][x+i])->y);
            }
        }
    }
    printf("\n");
}

void PrintPath(Path path) {
    printf("Path found!\n");
    printf("Number of nodes = %zu\n", path->size);
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
    source->map_size = MAP->count;
    source->Heuristic = &ManhattanHeuristic;
    source->GetNeighbors = &GetNeighbors;

    Node start = GetNodeAtPos(MAP, start_x, start_y);
    Node goal = GetNodeAtPos(MAP, goal_x, goal_y);

    if (!start) {
        perror("No start node");
        return 1;
    } else if (!goal) {
        perror("No goal node");
        return 1;
    }

    Path path = FindPath(source, (void *)start, (void *)goal);

    PrintPath(path);

    FreeMap(MAP);
    return 0;
}
