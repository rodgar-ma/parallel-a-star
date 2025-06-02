#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include "heap.h"

static inline void swap(heap_t *heap, int i, int j) {
    heap_item_t tmp_item;
    heap->items[i].node = heap->items[i].node;
    heap->items[i].fCost = heap->items[i].fCost;
    heap->items[i] = heap->items[j];
    heap->items[j].node = tmp_item.node;
    heap->items[j].fCost = tmp_item.fCost;
}

heap_t *heap_init(void) {
    heap_t *heap = malloc(sizeof(heap_t));
    heap->size = 0;
    heap->capacity = INIT_CAPACITY;
    heap->items = malloc(INIT_CAPACITY * sizeof(heap_item_t));
    return heap;
}

void heap_destroy(heap_t *heap) {
    free(heap->items);
    free(heap);
}

void heap_insert(heap_t *heap, int n_id, float fCost) {
    heap->size++;
    if (heap->size == heap->capacity) {
        heap->capacity *= 2;
        heap->items = realloc(heap->items, heap->capacity * sizeof(heap_item_t));
    }
    heap->items[heap->size].node = n_id;
    heap->items[heap->size].fCost = fCost;
    int current = heap->size;
    while (current > 1 && heap->items[current].fCost < heap->items[current / 2].fCost) {
        swap(heap, current, current / 2);
        current = current / 2;
    }
}

heap_item_t heap_extract(heap_t *heap) {
    swap(heap, 1, heap->size);
    heap_item_t res;
    res.node = heap->items[heap->size].node;
    res.fCost = heap->items[heap->size].fCost;
    heap->size--;
    int current = 1;
    while (current < heap->size) {
        int smallest = current;
        int child = 2 * current;
        if (child <= heap->size && heap->items[child].fCost < heap->items[smallest].fCost) {
            smallest = child;
        }
        child = 2 * current + 1;
        if (child <= heap->size && heap->items[child].fCost < heap->items[smallest].fCost) {
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