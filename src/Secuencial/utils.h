#ifndef UTILS_H
#define UTILS_H

#include "MapReader.h"
#include "astar.h"

Map MAP;

Node GetNodeById(Map map, unsigned long id);

int ExistsNodeAtPos(Map map, int x, int y);

unsigned long GetIdAtPos(Map map, int x, int y);

double ChevyshevHeuristic(astar_id_t n1_id, astar_id_t n2_id);

double MahattanHeuristic(astar_id_t n1_id, astar_id_t n2_id);

double DiagonalHeuristic(astar_id_t n1_id, astar_id_t n2_id);

void GetNeighbors8Tiles(neighbors_list *neighbors, astar_id_t n_id);

void GetNeighbors4Tiles(neighbors_list *neighbors, astar_id_t n_id);

void GetNeighbors(neighbors_list *neighbors, astar_id_t n_id);

void PrintPath(path *path);

#endif