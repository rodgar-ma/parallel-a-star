#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include "heap.h"

static inline void swap(heap_t *heap, int i, int j) {
    node_t *tmp = heap->nodes[i];
    heap->nodes[i] = heap->nodes[j];
    heap->nodes[j] = tmp;
}

heap_t *heap_init(void) {
    heap_t *heap = malloc(sizeof(heap_t));
    heap->size = 0;
    heap->capacity = INIT_CAPACITY;
    heap->nodes = malloc(INIT_CAPACITY * sizeof(node_t*));
    return heap;
}

void heap_destroy(heap_t *heap) {
    free(heap->nodes);
    free(heap);
}

void heap_insert(heap_t *heap, node_t *n) {
    heap->size++;
    if (heap->size == heap->capacity) {
        heap->capacity *= 2;
        heap->nodes = realloc(heap->nodes, heap->capacity * sizeof(node_t*));
    }
    heap->nodes[heap->size] = n;
    int current = heap->size;
    while (current > 1 && heap->nodes[current]->fCost < heap->nodes[current / 2]->fCost) {
        swap(heap, current, current / 2);
        current = current / 2;
    }
}

node_t *heap_extract(heap_t *heap) {
    if (heap->size == 0) return NULL;
    swap(heap, 1, heap->size);
    node_t *res = heap->nodes[heap->size--];
    int current = 1;
    while (current < heap->size) {
        int smallest = current;
        int child = 2 * current;
        if (child <= heap->size && heap->nodes[child]->fCost < heap->nodes[smallest]->fCost) {
            smallest = child;
        }
        child = 2 * current + 1;
        if (child <= heap->size && heap->nodes[child]->fCost < heap->nodes[smallest]->fCost) {
            smallest = child;
        }
        if (current == smallest) break;
        swap(heap, current, smallest);
        current = smallest;
    }
    return res;
}

int heap_is_empty(heap_t *heap) {
    return heap->size == 0;
}
