#ifndef HEAP_H
#define HEAP_H

#include "astar.h"

#define INIT_CAPACITY 1000

typedef struct heap_item_t {
    int node;
    float fCost;
} heap_item_t;

typedef struct heap_t {
    int size;
    int capacity;
    heap_item_t *items;
} heap_t;

heap_t *heap_init(void);

void heap_destroy(heap_t *heap);

heap_item_t *heap_extract(heap_t *heap);

void heap_insert(heap_t *heap, int n_id, float fCost);

int heap_is_empty(heap_t *heap);


#endif