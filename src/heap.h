#ifndef HEAP_H
#define HEAP_H

#include "astar.h"

#define INIT_CAPACITY 1000

typedef struct heap_t {
    int size;
    int capacity;
    node_t **nodes;
} heap_t;

heap_t *heap_init(void);

void heap_destroy(heap_t *heap);

node_t *heap_extract(heap_t *heap);

void heap_insert(heap_t *heap, node_t *node);

void heap_update(heap_t *heap, node_t *node);

int heap_is_empty(heap_t *heap);

float heap_min(heap_t *heap);


#endif