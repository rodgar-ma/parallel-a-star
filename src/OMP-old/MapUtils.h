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

Node GetNodeById(Map map, astar_id_t id);
int ExistsNodeAtPos(Map map, int x, int y);
astar_id_t GetIdAtPos(Map map, int x, int y);

double ChevyshevHeuristic(astar_id_t n1_id, astar_id_t n2_id);
double MahattanHeuristic(astar_id_t n1_id, astar_id_t n2_id);
double DiagonalHeuristic(astar_id_t n1_id, astar_id_t n2_id);
double EuclideanHeuristic(astar_id_t n1_id, astar_id_t n2_id);

void GetNeighbors8Tiles(neighbors_list *neighbors, astar_id_t n_id);
void GetNeighbors4Tiles(neighbors_list *neighbors, astar_id_t n_id);
void GetNeighbors(neighbors_list *neighbors, astar_id_t n_id);

void PrintPath(path *path);

#endif