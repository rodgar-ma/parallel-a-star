#include <stdlib.h>
#include "AStar.h"

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

struct __HashItem{
    Node node;
    HashItem next;
};

struct __HashTable {
    size_t capacity;
    size_t count;
    HashItem *nodes;
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
        open->nodes = realloc(open->nodes, open->capacity * sizeof(Node *));
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
    ht->capacity = 0;
    ht->count = 0;
    ht->nodes = NULL;
    return ht;
}

size_t HashFunction(void *node, size_t capacity) {
    return ((size_t)node) % capacity;
}

void ResizeHashTable(HashTable visited) {
    size_t newCapacity = (visited->capacity == 0) ? 16 : visited->capacity * 2;
    HashItem *newNodes = calloc(newCapacity, sizeof(struct __HashItem));

    // Reinsertar nodos en la nueva tabla
    for (size_t i = 0; i < visited->capacity; i++) {
        HashItem current = visited->nodes[i];
        while (current) {
            HashItem next = current->next; // Guardamos el siguiente nodo en la lista de colisiones
            size_t newIndex = HashFunction(current->node, newCapacity);
            
            // Insertar en la nueva tabla con manejo de colisiones
            current->next = newNodes[newIndex];
            newNodes[newIndex] = current;

            current = next;
        }
    }

    // Liberar la tabla anterior y actualizar la referencia
    free(visited->nodes);
    visited->nodes = newNodes;
    visited->capacity = newCapacity;
}

void FreeHashTable(HashTable visited) {
    if (!visited) return;

    for (size_t i = 0; i < visited->capacity; i++) {
        HashItem current = visited->nodes[i];
        while (current) {
            HashItem next = current->next;
            free(current->node);
            free(current);
            current = next;
        }
    }

    free(visited->nodes);
    free(visited);
}

Node GetNode(HashTable visited, void *node) {
    if (visited->capacity == 0 || (visited->count + 1) > visited->capacity * 0.75) {
        ResizeHashTable(visited);
    }

    size_t index = HashFunction(node, visited->capacity);
    HashItem current = visited->nodes[index];

    while (current) {
        if (current->node == node) {
            return current->node;
        }
        current = current->next;
    }

    HashItem hi = malloc(sizeof(struct __HashItem));
    hi->node = CreateNode(node);
    hi->next = visited->nodes[index];
    visited->nodes[index] = hi;
    visited->count++;

    return hi->node;
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

void AddNodeToVisitedList(Node node) {
    node->isClosed = 1;
    return;
}

Path FindPath(AStarSource *source, void *start, void *goal) {
    PriorityQueue open = CreatePriorityQueue();
    HashTable visited = CreateHashTable();
    NeighborsList neighborsList = CreateNeighborsList();
    Path path = NULL;

    Node current = GetNode(visited, start);
    current->gCost = 0;
    current->fCost = source->Heuristic(start, goal);

    AddNodeToOpenList(open, current);

    while(HasOpenNodes(open)) {
        current = GetFirstFromOpen(open);
        if (current->node == goal) {
            AddNodeToVisitedList(current);
            break;
        }
        
        AddNodeToVisitedList(current);
        
        neighborsList->count = 0;
        source->GetNeighbors(neighborsList, current->node);
        for (size_t i = 0; i < neighborsList->count; i++) {
            Node neighbor = GetNode(visited, neighborsList->nodes[i]);
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
    FreeHashTable(visited);
    FreeNeighborsList(neighborsList);
    return path;
}
