#ifndef MAP_READER_H
#define MAP_READER_H

typedef struct __Node *Node;
typedef struct __Map *Map;

struct __Node {
    unsigned long id;
    int x;
    int y;
};

struct __Map {
    int count;
    int width;
    int height;
    Node** grid;
};

Map LoadMap(char *filename);

void FreeMap(Map map);

#endif