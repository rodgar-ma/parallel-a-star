#include <stdlib.h>
#include <stdio.h>

// Forward declarations
typedef struct __Node *Node;
typedef struct __NeighborsList *NeighborsList;
typedef struct __NeighborRecord *NeighborRecord;
typedef struct __OpenSet *OpenSet;
typedef struct __ClosedSet *ClosedSet;

struct __Node {
    float h_value;
    float g_value;
    unsigned isOpen:1;
    unsigned isClosed:1;
    NeighborsList neighbors;
};

struct __NeighborsList {
    float cost;
    NeighborRecord nodes;
    NeighborsList next;
};

struct __NeighborRecord {
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
    return n;
}

NeighborRecord CreateNodeRecord(Node n) {
    NeighborRecord nr = malloc(sizeof(NeighborRecord));
    nr->node = n;
    return nr;
}

NeighborsList CreateNeighborsList(float cost) {
    NeighborsList nl = malloc(sizeof(NeighborsList));
    nl->cost = cost;
    return nl;
}

void AddNeighborsList(Node n, NeighborsList neighbors) {
    NeighborsList list = n->neighbors;
    while(list != NULL) {
        if (list->cost > neighbors->cost) break;
        list++;
    }
    if (list != NULL) {
        NeighborsList tmp = list;
        list = neighbors;
        list->next = tmp;
    } else {
        list = neighbors;
    }
}

NeighborsList GetNeighborsOfCost(Node n, float cost) {
    NeighborsList list = n->neighbors;
    while (list != NULL) {
        if (list->cost == cost) break;
        list++;
    }
    return list;
}

void AddRecord(NeighborsList neighbors, NeighborRecord nr) {
    nr->next = neighbors->nodes;
    neighbors->nodes = nr;
}

void AddNeighbor(Node n, Node new, float cost)
{   
    NeighborRecord nr = CreateNodeRecord(new);
    NeighborsList neighbors = GetNeighborsOfCost(n, cost);
    if (neighbors == NULL) {
        neighbors = CreateNeighborsList(cost);
    }
    AddRecord(neighbors, nr);
    AddNeighborsList(n, neighbors);
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
    
    return 0;
}

