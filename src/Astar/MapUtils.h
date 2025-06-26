#ifndef MAP_UTILS_H
#define MAP_UTILS_H

#include "astar.h"

typedef struct {
    unsigned long id;
    int x;
    int y;
    int walkable:1;
} Node;

typedef struct {
    int width;
    int height;
    Node*** grid;
} Map;

extern Map *MAP;

Map *LoadMap(char *filename);
void FreeMap(Map *map);

void idToXY(int id, int *x, int *y);
int xyToID(int x, int y);

int inrange(int x, int y);
int is_walkable(int x, int y);

float ChevyshevHeuristic(int n1_id, int n2_id);
float ManhattanHeuristic(int n1_id, int n2_id);
float DiagonalHeuristic(int n1_id, int n2_id);

void GetNeighbors(neighbors_list *neighbors, int n_id);
void GetFourNeighbors(neighbors_list *neighbors, int n_id);

void PrintPath(path *path);

#endif