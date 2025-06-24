#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <stdio.h>
#include "heap.h"

static inline void swap(heap_t *heap, int i, int j) {
    node_t *tmp = heap->nodes[i];
    heap->nodes[i] = heap->nodes[j];
    heap->nodes[j] = tmp;
    heap->nodes[i]->open_index = i;
    heap->nodes[j]->open_index = j;
}

heap_t *heap_init(void) {
    heap_t *heap = malloc(sizeof(heap_t));
    heap->size = 0;
    heap->capacity = INIT_CAPACITY;
    heap->nodes = malloc(heap->capacity * sizeof(node_t *));
    return heap;
}

void heap_destroy(heap_t *heap) {
    free(heap->nodes);
    free(heap);
}

void heap_insert(heap_t *heap, node_t *node) {
    if (heap->size == heap->capacity) {
        heap->capacity *= 2;
        heap->nodes = realloc(heap->nodes, heap->capacity * sizeof(node_t *));
    }

    int pos = heap->size++;
    heap->nodes[pos] = node;
    node->is_open = 1;
    node->open_index = pos;

    // Bubble up
    while (pos > 0) {
        int parent = (pos - 1) / 2;
        if (heap->nodes[pos]->fCost >= heap->nodes[parent]->fCost) break;
        swap(heap, pos, parent);
        pos = parent;
    }
}

node_t *heap_extract(heap_t *heap) {
    if (heap->size == 0) return NULL;

    node_t *min = heap->nodes[0];
    heap->size--;

    if (heap->size > 0) {
        heap->nodes[0] = heap->nodes[heap->size];
        heap->nodes[0]->open_index = 0;

        // Bubble down
        int pos = 0;
        int steps = 0;
        while (1) {
            steps++;
            int left = 2 * pos + 1;
            int right = 2 * pos + 2;
            int smallest = pos;

            if (left < heap->size && heap->nodes[left]->fCost < heap->nodes[smallest]->fCost)
                smallest = left;
            if (right < heap->size && heap->nodes[right]->fCost < heap->nodes[smallest]->fCost)
                smallest = right;

            if (smallest == pos) break;
            swap(heap, pos, smallest);
            pos = smallest;
        }
    }

    min->is_open = 0;
    min->open_index = -1;
    return min;
}

void heap_update(heap_t *heap, node_t *node) {
    int pos = node->open_index;

    // Bubble up
    while (pos > 0) {
        int parent = (pos - 1) / 2;
        if (heap->nodes[pos]->fCost >= heap->nodes[parent]->fCost) break;
        swap(heap, pos, parent);
        pos = parent;
    }
    
    int steps = 0;
    // Bubble down
    while (1) {
        steps++;
        int left = 2 * pos + 1;
        int right = 2 * pos + 2;
        int smallest = pos;

        if (left == -1) {
            printf("\n");
        }

        if (left < heap->size && heap->nodes[left]->fCost < heap->nodes[smallest]->fCost)
            smallest = left;
        if (right < heap->size && heap->nodes[right]->fCost < heap->nodes[smallest]->fCost)
            smallest = right;

        if (smallest == pos) break;
        swap(heap, pos, smallest);
        pos = smallest;
    }
}

int heap_is_empty(heap_t *heap) {
    return heap->size == 0;
}

float heap_min(heap_t *heap) {
    if (heap == NULL || heap->size < 0 || heap->nodes == NULL || heap->nodes[0] == NULL)
        return FLT_MAX;
    return heap->nodes[0]->fCost;
}