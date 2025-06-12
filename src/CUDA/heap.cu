#include <stdlib.h>
#include <limits.h>
#include <cuda.h>
#include "heap.h"

__device__ static inline void swap(heap_t *heap, int i, int j) {
    node_t *tmp = heap->nodes[i];
    heap->nodes[i] = heap->nodes[j];
    heap->nodes[j] = tmp;
}

heap_t *heap_init(void) {
    heap_t heap_cpu;
    heap_t *heap_gpu;
    heap_cpu.size = 0;
    
    return heap_gpu;
}

heap_t **heaps_init(int k) {
    heap_t **heaps = malloc(k * sizeof(heap_t *));
    for (int i = 0; i < k; i++) {
        heaps[i] = heap_init();
    }
    return heaps;
}

void heap_destroy(heap_t *heap) {
    free(heap->nodes);
    free(heap);
}

void heaps_destroy(heap_t **heaps, int k) {
    for (int i = 0; i < k; i++) {
        heap_destroy(heaps[i]);
    }
    free(heaps);
}

void heap_insert(heap_t *heap, node_t *node) {
    heap->size++;
    if (heap->size == heap->capacity) {
        heap->capacity *= 2;
        heap->nodes = realloc(heap->nodes, heap->capacity * sizeof(node_t *));
    }
    heap->nodes[heap->size] = node;
    node->is_open = 1;
    node->open_index = heap->size;
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
    res->is_open = 0;
    res->open_index = -1;
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

void heap_update(heap_t *heap, node_t *node) {
    int position = node->open_index;
    while(position > 1 && heap->nodes[position]->fCost < heap->nodes[position / 2]->fCost) {
        swap(heap, position, position / 2);
        position = position / 2;
    }
}

int heap_is_empty(heap_t *heap) {
    return heap->size == 0;
}
