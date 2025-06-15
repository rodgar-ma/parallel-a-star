#ifndef HEAP_H
#define HEAP_H

#include "astar.h"

#define INIT_CAPACITY 16 * 1024

struct heap_t {
    int size;
    int capacity;
    node_t **nodes;
};

heap_t *heap_init(void);

heap_t **heaps_init(int k);

void heap_destroy(heap_t *heap);

void heaps_destroy(heap_t **heaps, int k);

__device__ node_t *heap_extract(heap_t *heap);

__device__ void heap_insert(heap_t *heap, node_t *node);

__device__ void heap_update(heap_t *heap, node_t *node);

__device__ int heap_is_empty(heap_t *heap);


#endif