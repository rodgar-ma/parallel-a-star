#ifndef LIST_H
#define LIST_H

#include "astar.h"

struct list_t {
    int size;
    int capacity;
    node_t **nodes;
};

list_t **lists_create(int lists, int capacity);

list_t *list_create(int capacity);

void lists_destroy(list_t **lists_gpu, int lists);

void list_destroy(list_t *list);

__device__ void list_clear(list_t *list);

__device__ void list_insert(list_t *list, node_t *node);

__device__ void list_remove(list_t *list, int index);

__device__ node_t *list_get(list_t *list, int index);

#endif