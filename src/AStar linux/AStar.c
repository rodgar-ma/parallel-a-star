#include <stdlib.h>
#include "AStar.h"

#define MAP_SIZE 262144

typedef struct __Node *Node;
typedef struct __PriorityQueue *PriorityQueue;
typedef struct __HashItem *HashItem;
typedef struct __HashTable *HashTable;

struct __NeighborsList {
    size_t capacity;
    size_t count;
    int *costs;
    void **nodes;
};

struct __Node {
    void *node;
    Node parent;
    int gCost;
    int fCost;
    unsigned isOpen:1;
    unsigned isClosed:1;
};

struct __PriorityQueue {
    size_t capacity;
    size_t count;
    Node *nodes;
} ;

struct __HashTable {
    size_t count;
    Node *nodes;
};

NeighborsList CreateNeighborsList() {
    NeighborsList list = malloc(sizeof(struct __NeighborsList));
    list->capacity = 0;
    list->count = 0;
    list->costs = NULL;
    list->nodes = NULL;
    return list;
}

Node CreateNode(void *n) {
    Node node = malloc(sizeof(struct __Node));
    node->node = n;
    node->parent = NULL;
    node->gCost = 0;
    node->fCost = 0;
    node->isOpen = 0;
    node->isClosed = 0;
    return node;
}

void AddNeighbor(NeighborsList list, void *node, int cost) {
    if (list->count == list->capacity) {
        list->capacity = 1 + (2 * list->capacity);
        list->costs = realloc(list->costs, list->capacity * sizeof(int));
        list->nodes = realloc(list->nodes, list->capacity * sizeof(void *));
    }
    list->costs[list->count] = cost;
    list->nodes[list->count] = node;
    list->count++;
}

 void FreeNeighborsList(NeighborsList neighbors) {
    free(neighbors->costs);
    free(neighbors->nodes);
    free(neighbors);
 }

PriorityQueue CreatePriorityQueue() {
    PriorityQueue pq = malloc(sizeof(struct __PriorityQueue));
    pq->capacity = 0;
    pq->count = 0;
    pq->nodes = NULL;
    return pq;
}

void Swap(Node *a, Node *b) {
    Node temp = *a;
    *a = *b;
    *b = temp;
}

void AddNodeToOpenList(PriorityQueue open, Node n) {
    n->isClosed = 0;
    if (n->isOpen) return;

    n->isOpen = 1;

    if (open->count == open->capacity) {
        open->capacity = 1 + (2 * open->capacity);
        open->nodes = realloc(open->nodes, open->capacity * sizeof(Node));
    }

    size_t i = open->count++;
    open->nodes[i] = n;

    while (i > 0 && open->nodes[(i-1)/2]->fCost > open->nodes[i]->fCost) {
        Swap(&open->nodes[i], &open->nodes[(i-1)/2]);
        i = (i - 1) / 2;
    }
}

Node GetFirstFromOpen(PriorityQueue open) {
    if (open->count == 0) return NULL;

    Node minNode = open->nodes[0];
    open->nodes[0] = open->nodes[--open->count];

    size_t i = 0;
    while (2 * i + 1 < open->count) {
        size_t left = 2 * i + 1;
        size_t right = 2 * i + 2;
        size_t smallest = left;

        if (right < open->count && open->nodes[right]->fCost < open->nodes[left]->fCost) {
            smallest = right;
        }

        if (open->nodes[i]->fCost <= open->nodes[smallest]->fCost) break;

        Swap(&open->nodes[i], &open->nodes[smallest]);
        i = smallest;
    }

    return minNode;
}

int HasOpenNodes(PriorityQueue pq) {
    return pq->count > 0;
}

void FreePriorityQueue(PriorityQueue pq) {
    free(pq->nodes);
    free(pq);
}

HashTable CreateHashTable() {
    HashTable ht = malloc(sizeof(struct __HashTable));
    ht->count = 0;
    ht->nodes = calloc(MAP_SIZE, sizeof(Node));
    return ht;
}

size_t HashFunction(void *node) {
    return ((size_t)node) % MAP_SIZE;
}

void FreeHashTable(HashTable visited) {
    free(visited->nodes);
    free(visited);
}

Node GetNode(HashTable visited, void *node) {
    size_t index = HashFunction(node);
    if (visited->nodes[index]) {
        return visited->nodes[index];
    }
    visited->nodes[index] = CreateNode(node);
    return visited->nodes[index];
}

Path RetracePath(Node goal) {
    Path path = malloc(sizeof(struct __Path));
    path->cost = goal->gCost;
    path->size = 0;
    Node current = goal;
    while (current) {
        path->size++;
        current = current->parent;
    }

    current = goal;
    path->nodes = calloc(path->size, sizeof(void *));
    for (size_t i = 0; i < path->size; i++) {
        path->nodes[i] = current->node;
        current = current->parent;
    }
    return path;
}

void AddNodeToClosedList(Node node) {
    node->isClosed = 1;
    return;
}

Path FindPath(AStarSource *source, void *start, void *goal) {
    PriorityQueue open = CreatePriorityQueue();
    HashTable hashTable = CreateHashTable();
    NeighborsList neighborsList = CreateNeighborsList();
    Path path = NULL;

    Node current = GetNode(hashTable, start);
    current->gCost = 0;
    current->fCost = source->Heuristic(start, goal);

    AddNodeToOpenList(open, current);

    while(HasOpenNodes(open)) {
        current = GetFirstFromOpen(open);
        if (current->node == goal) {
            AddNodeToClosedList(current);
            break;
        }
        
        AddNodeToClosedList(current);
        
        neighborsList->count = 0;
        source->GetNeighbors(neighborsList, current->node);
        for (size_t i = 0; i < neighborsList->count; i++) {
            Node neighbor = GetNode(hashTable, neighborsList->nodes[i]);
            int newCost = current->gCost + neighborsList->costs[i];
            if (newCost < neighbor->gCost || !neighbor->isOpen) {
                neighbor->gCost = newCost;
                neighbor->fCost = newCost + source->Heuristic(neighbor->node, goal);
                neighbor->parent = current;
                if (!neighbor->isOpen) {
                    AddNodeToOpenList(open, neighbor);
                }
            }
        }
    }
    path = RetracePath(current);
    FreePriorityQueue(open);
    FreeHashTable(hashTable);
    FreeNeighborsList(neighborsList);
    return path;
}
