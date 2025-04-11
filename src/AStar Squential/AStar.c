#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include "astar.h"

struct __NeighborsList {
    size_t capacity;
    size_t count;
    double *costs;
    void **nodes;
};

struct __node {
    void *node;
    node parent;
    double gCost;
    double fCost;
    unsigned isOpen:1;
    unsigned isClosed:1;
};

struct __PriorityQueue {
    size_t capacity;
    size_t count;
    node *nodes;
} ;

struct __HashTable {
    AStarSource source;
    size_t size;
    size_t count;
    node *nodes;
};

NeighborsList CreateNeighborsList() {
    NeighborsList list = malloc(sizeof(struct __NeighborsList));
    list->capacity = 0;
    list->count = 0;
    list->costs = NULL;
    list->nodes = NULL;
    return list;
}

node Createnode(void *n) {
    node node = malloc(sizeof(struct __node));
    node->node = n;
    node->parent = NULL;
    node->gCost = DBL_MAX;
    node->fCost = DBL_MAX;
    node->isOpen = 0;
    node->isClosed = 0;
    return node;
}

void AddNeighbor(NeighborsList list, void *node, double cost) {
    if (list->count == list->capacity) {
        list->capacity = 1 + (2 * list->capacity);
        list->costs = realloc(list->costs, list->capacity * sizeof(double));
        list->nodes = realloc(list->nodes, list->capacity * sizeof(void *));
    }
    list->costs[list->count] = cost;
    list->nodes[list->count] = node;
    list->count++;
}

void FreeNeighborsList(NeighborsList neighbors) {
    free(neighbors->costs);
    free(neighbors);
}

Path RetracePath(node goal) {
    Path path = malloc(sizeof(struct __Path));
    path->cost = goal->gCost;
    path->size = 0;
    node current = goal;
    while (current) {
        path->size++;
        current = current->parent;
    }

    current = goal;
    path->nodes = calloc(path->size, sizeof(void *));
    for (size_t i = 0; i < path->size; i++) {
        path->nodes[path->size-i-1] = current->node;
        current = current->parent;
    }
    return path;
}

void FreePath(Path path) {
    free(path->nodes);
    free(path);
}

Path FindPath(AStarSource source, void *start, void *goal) {
    PriorityQueue open = CreatePriorityQueue();
    HashTable hashTable = CreateHashTable(source);
    NeighborsList neighborsList = CreateNeighborsList();
    Path path = NULL;

    node current = Getnode(hashTable, start);
    current->gCost = 0;
    current->fCost = source.Heuristic(start, goal);

    AddnodeToOpenList(open, current);

    while(HasOpennodes(open)) {
        current = GetFirstFromOpen(open);
        current->isClosed = 1;
        if (current->node == goal) {
            break;
        }
        
        neighborsList->count = 0;
        source.GetNeighbors(neighborsList, current->node);
        for (size_t i = 0; i < neighborsList->count; i++) {
            node neighbor = Getnode(hashTable, neighborsList->nodes[i]);
            double newCost = current->gCost + neighborsList->costs[i];
            if (newCost < neighbor->gCost) {
                neighbor->gCost = newCost;
                neighbor->fCost = newCost + source.Heuristic(neighbor->node, goal);
                neighbor->parent = current;
                if (!neighbor->isOpen) {
                    AddnodeToOpenList(open, neighbor);
                } else {
                    ReorderOpenList(open, neighbor);
                }
            }
        }
    }
    path = RetracePath(current);
    FreeHashTable(hashTable);
    FreePriorityQueue(open);
    FreeNeighborsList(neighborsList);
    return path;
}
