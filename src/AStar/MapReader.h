#ifndef MapReader_h
#define MapReader_h

typedef struct __Node {
    int x;
    int y;
};

typedef struct __Node *Node;

typedef struct {
    int width;
    int height;
    Node ** grid;
} Map;

Map *LoadMap(const char *filename);
void FreeMap(Map *map);


#endif