#ifndef MAP_UTILS_H
#define MAP_UTILS_H

#include "astar.h"
#include "graph.h"

// typedef struct __Node *Node;
// typedef struct __Map *Map;

// struct __Node {
//     unsigned long id;
//     int x;
//     int y;
// };

// struct __Map {
//     int width;
//     int height;
//     Node** grid;
// };

// extern Map MAP;

Graph *LoadMap(char *filename);
void FreeMap(Graph *map);

Node GetNodeById(Graph *map, int id);
int ExistsNodeAtPos(Graph *map, int x, int y);
int GetIdAtPos(Graph *map, int x, int y);

float ChevyshevHeuristic(int n1_id, int n2_id);
float MahattanHeuristic(int n1_id, int n2_id);
float DiagonalHeuristic(int n1_id, int n2_id);

void GetNeighbors8Tiles(neighbors_list *neighbors, int n_id);
void GetNeighbors4Tiles(neighbors_list *neighbors, int n_id);
void GetNeighbors(neighbors_list *neighbors, int n_id);

void PrintPath(path *path);

#endif