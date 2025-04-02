#include <stdlib.h>
#include "AStar.h"

typedef struct {
    size_t capacity;
    size_t count;
    int *costs;
    void **nodes;
} NeighborsList;

typedef struct {
    void *node;
    Node *parent;
    int gCost;
    int fCost;
    unsigned isOpen:1;
    unsigned isClosed:1;
} Node;

typedef struct {
    size_t capacity;
    size_t count;
    Node **nodes;
} PriorityQueue;

typedef struct {
    Node *node;
    HashItem *next;
} HashItem;

typedef struct {
    size_t capacity;
    size_t count;
    HashItem **nodes;
} HashTable;

NeighborsList *CreateNeighborsList() {
    NeighborsList *list = malloc(sizeof(NeighborsList));
    list->capacity = 0;
    list->costs = 0;
    return list;
}

Node *CreateNode(void *n) {
    Node *node = malloc(sizeof(Node));
    node->node = n;
    node->isOpen = 0;
    node->isClosed = 0;
    return node;
}

void AddNeighbor(NeighborsList *list, void *node, int cost) {
    if (list->count == list->capacity) {
        list->capacity = 1 + (2 * list->capacity);
        list->costs = realloc(list->costs, list->capacity * sizeof(int));
        list->nodes = realloc(list->nodes, list->capacity * sizeof(void *));
    }
    list->costs[list->count] = cost;
    list->nodes[list->count] = node;
    list->count++;
}

PriorityQueue *CreatePriorityQueue() {
    PriorityQueue *pq = malloc(sizeof(PriorityQueue));
    pq->capacity = 0;
    pq->count = 0;
    pq->nodes = NULL;
    return pq;
}

void Swap(Node **a, Node **b) {
    Node *temp = *a;
    *a = *b;
    *b = temp;
}

void AddNodeToOpenList(PriorityQueue *open, Node *n) {
    if (n->isClosed) n->isClosed = 0;
    if (n->isOpen) return;

    n->isOpen = 1;

    if (open->count == open->capacity) {
        open->capacity = 1 + (2 * open->capacity);
        open->nodes = realloc(open->nodes, open->count * sizeof(Node *));
    }

    size_t i = open->count++;
    open->nodes[i] = n;

    while (i > 0 && open->nodes[(i-1)/2]->fCost > open->nodes[i]->fCost) {
        Swap(&open->nodes[i], &open->nodes[(i-1)/2]);
        i = (i - 1) / 2;
    }
}

Node *GetFirstFromOpen(PriorityQueue *open) {
    if (open->count == 0) return NULL;

    Node *minNode = open->nodes[0];
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

int HasOpenNodes(PriorityQueue *pq) {
    return pq->count > 0;
}

void DestroyPriorityQueue(PriorityQueue *pq) {
    free(pq->nodes);
    free(pq);
}

HashTable *CreateHashTable() {
    HashTable *ht = malloc(sizeof(HashTable));
    ht->capacity = 0;
    ht->count = 0;
    ht->nodes = NULL;
    return ht;
}

size_t HashFunction(void *node, size_t capacity) {
    return ((size_t)node) % capacity;
}

Node *GetNode(HashTable *visited, void *node) {
    if (visited->capacity == 0 || (visited->count + 1) > visited->capacity * 0.75) {
        ResizeHashTable(visited);
    }

    size_t index = HashFunction(node, visited->capacity);
    HashItem *current = visited->nodes[index];

    while (current) {
        if (current->node == node) {
            return current->node;
        }
        current = current->next;
    }

    HashItem *hi = malloc(sizeof(HashItem));
    hi->node = CreateNode(node);
    hi->next = visited->nodes[index];
    visited->nodes[index] = hi;
    visited->count++;

    return hi->node;
}

void ResizeHashTable(HashTable *visited) {
    size_t newCapacity = (visited->capacity == 0) ? 16 : visited->capacity * 2;
    HashItem **newNodes = calloc(newCapacity, sizeof(HashItem *));

    // Reinsertar nodos en la nueva tabla
    for (size_t i = 0; i < visited->capacity; i++) {
        HashItem *current = visited->nodes[i];
        while (current) {
            HashItem *next = current->next; // Guardamos el siguiente nodo en la lista de colisiones
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

void FindPath(AStarSource *source, void *start, void *goal) {
    PriorityQueue *open = CreatePriorityQueue();
    HashTable *visited = CreateHashTable();
    NeighborsList *neighborsList = CreateNeighborsList();

    Node *current = GetNode(visited, start);
    current->gCost = 0;
    current->fCost = source->Heuristic(start, goal);

    AddNodeToOpenList(open, current);

    while(HasOpenNodes(open)) {
        Node *current = GetFirstFromOpen(open);
        if (current->node == goal) {
            return;
        }
        AddNodeToVisitedList(visited, current);
        source->GetNeighbors(neighborsList, current->node);
        for (size_t i = 0; i < neighborsList->count; i++) {
            Node *neighbor = GetNode(visited, neighborsList->nodes[i]);
            int newCost = current->gCost + neighborsList->costs[i];
            if (newCost < neighbor->gCost || !neighbor->isOpen) {
                neighbor->gCost = newCost;
                neighbor->fCost = newCost + source->Heuristic(neighbor->node, goal);
                if (!neighbor->isOpen) {
                    AddNodeToOpenList(open, neighbor);
                }
            }
        }
    }
}
