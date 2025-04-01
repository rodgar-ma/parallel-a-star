#ifndef MapReader_h
#define MapReader_h

typedef struct __Node *Node;
typedef struct __Map *Map;

typedef struct __Node {
    int x;
    int y;
};

typedef struct __Node *Node;

typedef struct __Map {
    int width;
    int height;
    Node ** grid;
};

Map LoadMap(const char *filename);
void FreeMap(Map map);


#endif