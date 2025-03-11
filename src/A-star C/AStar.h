#ifndef AStar_h
#define AStar_h

#include <stddef.h>

// Forward declarations
typedef struct __Node Node;
typedef struct __NeighborRecord NeighborRecord;
typedef struct __NeighborsList NeighborsList;
typedef struct __OpenSet OpenSet;
typedef struct __ClosedSet ClosedSet;

// Complete struct definitions
struct __NeighborRecord {
    float cost;
    Node* node;
};

struct __NeighborsList {
    size_t capacity;
    size_t count;
    NeighborRecord* list;
};

struct __Node {
    size_t nodeSize;
    float h_value;
    float g_value;
    unsigned isOpen:1;
    unsigned isClosed:1;
    NeighborsList neighbors;
};

// Function declarations
Node CreateNode(float hValue, float gValue);
void AddNeighbor(Node* n, Node* neighbor, float cost);

#endif