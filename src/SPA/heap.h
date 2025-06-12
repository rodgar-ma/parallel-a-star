#ifndef HEAP_H
#define HEAP_H

#include <omp.h>
#include "astar.h"

#define INIT_CAPACITY 1000000000

typedef struct heap_t {
    int size;
    int capacity;
    node_t **nodes;
    omp_lock_t lock;
} heap_t;

heap_t *heap_init(void);

void heap_destroy(heap_t *heap);

node_t *heap_extract(heap_t *heap);

void heap_insert(heap_t *heap, node_t *n);

int heap_is_empty(heap_t *heap);

float heap_min(heap_t *heap);


#endif