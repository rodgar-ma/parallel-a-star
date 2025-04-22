#ifndef MAP_READER_H
#define MAP_READER_H

#include "astar.h"

typedef struct __Node *Node;
typedef struct __Map *Map;

struct __Node {
    id_t id;
    int x;
    int y;
};

struct __Map {
    int count;
    int width;
    int height;
    Node ** grid;
};

Map LoadMap(char *filename);

Node GetNodeById(Map map, id_t id);

int ExistsNodeAtPos(Map map, int x, int y);

id_t GetIdAtPos(Map map, int x, int y);

void FreeMap(Map map);

#endif