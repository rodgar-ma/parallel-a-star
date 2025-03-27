#include <stdlib.h>
#include <stdio.h>

// Forward declarations
typedef struct __Node *Node;
typedef struct __NeighborsList *NeighborsList;
typedef struct __NeighborRecord *NeighborRecord;
typedef struct __OpenSet *OpenSet;
typedef struct __ClosedSet *ClosedSet;

static const NeighborRecord NullNeighbor = {NULL, -1};

struct __Node {
    float h_value;
    float g_value;
    unsigned isOpen:1;
    unsigned isClosed:1;
    NeighborsList neighbors;
};

struct __NeighborsList {
    int capacity;
    int count;
    NeighborRecord *list;
};

struct __NeighborRecord {
    float cost;
    Node node;
    NeighborRecord next;
};

Node CreateNode(float hValue, float gValue)
{
    Node n = malloc(sizeof(Node));
    n->h_value = hValue;
    n->g_value = gValue;
    n->isOpen = 0;
    n->isClosed = 0;
    n->neighbors = malloc(sizeof(NeighborsList));
    n->neighbors->capacity = 1;
    n->neighbors->count = 0;
    return n;
}

void DestroyNode(Node n) {
    free(n);
}

void AddNeighbor(Node n, Node neighbor, float cost)
{   
    NeighborRecord nr = malloc(sizeof(NeighborRecord));
    nr->cost = cost;
    nr->node = neighbor;

    if (n->neighbors->count == n->neighbors->capacity) {
        n->neighbors->capacity = 1 + (n->neighbors->capacity * 2);
        n->neighbors->list = realloc(n->neighbors->list, n->neighbors->capacity * sizeof(NeighborRecord));
    }

    int pos = 0;
    for (int i = 0; i < n->neighbors->count; i++) {
        if (n->neighbors->list[i]->cost <= nr->cost) {
            pos += 1;
        }
    }
    
    if (n->neighbors->list[pos] != NullNeighbor) {
        n->neighbors->list[pos]->next = nr;
    }
    nr->next = n->neighbors->list[pos+1];
    
}

int main(int argc, char const *argv[])
{
    Node n1 = CreateNode(1, 1);
    Node n2 = CreateNode(1, 1);
    Node n3 = CreateNode(1, 1);
    Node n4 = CreateNode(1, 1);
    Node n5 = CreateNode(1, 1);
    Node n6 = CreateNode(1, 1);
    
    AddNeighbor(n1, n2, 1);
    AddNeighbor(n1, n3, 2);
    AddNeighbor(n1, n4, 2);
    AddNeighbor(n1, n5, 3);
    AddNeighbor(n1, n6, 1);
    
    
    DestroyNode(n1);
    DestroyNode(n2);
    DestroyNode(n3);
    return 0;
}

