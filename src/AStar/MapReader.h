#ifndef MapReader_h
#define MapReader_h

typedef struct __Node *Node;
typedef struct __Map *Map;

struct __Node {
    int x;
    int y;
};

struct __Map {
    int width;
    int height;
    Node ** grid;
};

Map LoadMap(const char *filename);
Node GetNodeAtPos(Map map, int x, int y);
void FreeMap(Map map);


#endif