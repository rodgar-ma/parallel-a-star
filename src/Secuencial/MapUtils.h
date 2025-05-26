#ifndef MAP_UTILS_H
#define MAP_UTILS_H

#include "astar.h"

typedef struct __Node *Node;
typedef struct __Map *Map;

struct __Node {
    unsigned long id;
    int x;
    int y;
};

struct __Map {
    int width;
    int height;
    Node** grid;
};

extern Map MAP;

Map LoadMap(char *filename);
void FreeMap(Map map);

Node GetNodeById(Map map, int id);
int ExistsNodeAtPos(Map map, int x, int y);
int GetIdAtPos(Map map, int x, int y);

double ChevyshevHeuristic(int n1_id, int n2_id);
double MahattanHeuristic(int n1_id, int n2_id);
double DiagonalHeuristic(int n1_id, int n2_id);

void GetNeighbors8Tiles(neighbors_list *neighbors, int n_id);
void GetNeighbors4Tiles(neighbors_list *neighbors, int n_id);
void GetNeighbors(neighbors_list *neighbors, int n_id);

void PrintPath(path *path);

#endif