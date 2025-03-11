#include "AStar.h"
#include <stdlib.h>
#include <stdio.h>

typedef struct __Node Node;
typedef struct __NeighborsList NeighborsList;
typedef struct __NeighborRecord NeighborRecord;

Node CreateNode(float hValue, float gValue)
{
    Node n;
    n.nodeSize = sizeof(Node);
    n.h_value = hValue;
    n.g_value = gValue;
    n.isOpen = 0;
    n.isClosed = 0;
    n.neighbors.capacity = 1;
    n.neighbors.count = 0;
    n.neighbors.list = malloc(sizeof(NeighborRecord) * n.neighbors.capacity);
    return n;
}

void AddNeighbor(Node* n, Node* neighbor, float cost)
{
    NeighborRecord nr;
    nr.cost = cost;
    nr.node = neighbor;
    
    if (n->neighbors.count == n->neighbors.capacity) {
        n->neighbors.capacity = 1 + (n->neighbors.capacity * 2);
        n->neighbors.list = realloc(n->neighbors.list, n->neighbors.capacity * sizeof(NeighborRecord));
    }
    n->neighbors.list[n->neighbors.count++] = nr;
}

int main(int argc, char const *argv[])
{
    Node n1 = CreateNode(0, 2);
    Node n2 = CreateNode(1, 1);
    Node n3 = CreateNode(2, 0);
    printf("Node 1:\n");
    printf("  hValue = %f\n", n1.h_value);
    printf("  gValue = %f\n", n1.g_value);
    printf("  Node size = %zu\n", n1.nodeSize);
    printf("\nNode 2:\n");
    printf("  hValue = %f\n", n2.h_value);
    printf("  gValue = %f\n", n2.g_value);
    printf("  Node size = %zu\n", n2.nodeSize);
    printf("\nNode 3:\n");
    printf("  hValue = %f\n", n3.h_value);
    printf("  gValue = %f\n", n3.g_value);
    printf("  Node size = %zu\n", n3.nodeSize);
    
    AddNeighbor(&n1, &n2, 2);
    printf("\nAdded n2 as neighbor to n1 with cost %f\n", n1.neighbors.list[0].cost);
    AddNeighbor(&n2, &n3, 3);
    printf("\nAdded n3 as neighbor to n2 with cost %f\n", n2.neighbors.list[0].cost);
    
    // Free allocated memory
    free(n1.neighbors.list);
    free(n2.neighbors.list);
    free(n3.neighbors.list);
    return 0;
}
