
#include "AStar.h"
#include <stdlib.h>

typedef struct __Node *Node;
typedef struct __NeighborsList *NeighborsList;
typedef struct __OpenSet *OpenSet;

struct __Node
{
    size_t nodeSize;
    float h_value;
    float g_value;
    unsigned isOpen:1;
    unsigned isClosed:1;
    NeighborsList neighbors;
};

struct __NeighborsList
{
    size_t size;
    size_t count;
    Node* first;
};

struct __OpenSet
{
    size_t size;
    Node* next;
};

struct __ClosedSet
{
    size_t size;
    Node* first;
};

Node createNode(float h_value, float g_value)
{
    Node n;
    n->nodeSize = sizeof(Node);
    n->h_value = h_value;
    n->g_value = g_value;
}

void addNeighbor(Node* node, Node* neighbor)
{
    return;
}