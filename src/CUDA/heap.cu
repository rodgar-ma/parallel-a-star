#include <stdlib.h>
#include <limits.h>
#include <cuda.h>
#include "heap.h"
#include "cuda_utils.h"

__device__ static inline void swap(heap_t *heap, int i, int j) {
    node_t *tmp = heap->nodes[i];
    heap->nodes[i] = heap->nodes[j];
    heap->nodes[j] = tmp;
}

heap_t *heap_init(void) {
    heap_t heap_cpu;
    heap_t *heap_gpu;
    heap_cpu.size = 0;
    HANDLE_RESULT(cudaMalloc(&heap_cpu.nodes, (INIT_CAPACITY + 1) * sizeof(node_t*)));
    HANDLE_RESULT(cudaMemset(heap_cpu.nodes, 0, (INIT_CAPACITY + 1) * sizeof(node_t*)));

    HANDLE_RESULT(cudaMalloc(&heap_gpu, sizeof(heap_t)));
    HANDLE_RESULT(cudaMemcpy(heap_gpu, &heap_cpu, sizeof(heap_t), cudaMemcpyDefault));
    return heap_gpu;
}

heap_t **heaps_init(int k) {
    heap_t **heaps_cpu = (heap_t**)malloc(k * sizeof(heap_t*));
    heap_t **heap_gpu = NULL;
    for (int i = 0; i < k; i++) {
        heaps_cpu[i] = heap_init();
    }

    HANDLE_RESULT(cudaMalloc(&heap_gpu, k * sizeof(heap_t*)));
    HANDLE_RESULT(cudaMemcpy(heap_gpu, heaps_cpu, k * sizeof(heap_t*), cudaMemcpyDefault));
}

void heap_destroy(heap_t *heap) {
    heap_t heap_cpu;
    HANDLE_RESULT(cudaMemcpy(&heap_cpu, heap, sizeof(heap_t), cudaMemcpyDefault));
    HANDLE_RESULT(cudaFree(heap_cpu.nodes));
    HANDLE_RESULT(cudaFree(heap));
}

void heaps_destroy(heap_t **heaps, int k) {
    heap_t **heaps_cpu = (heap_t**)malloc(k * sizeof(heap_t*));
    HANDLE_RESULT(cudaMemcpy(heaps_cpu, heaps, k * sizeof(heap_t*), cudaMemcpyDefault));
    for (int i = 0; i < k; i++) {
        heap_destroy(heaps_cpu[i]);
    }
    free(heaps_cpu);
    HANDLE_RESULT(cudaFree(heaps));
}

__device__ void heap_insert(heap_t *heap, node_t *node) {
    heap->size++;
	heap->nodes[heap->size] = node;
	int current = heap->size;
	while (current > 1 && heap->nodes[current]->fCost < heap->nodes[current / 2]->fCost) {
		swap(heap, current, current / 2);
		current /= 2;
	}
}

__device__ node_t *heap_extract(heap_t *heap) {
    node_t *res = heap->nodes[1];
	heap->nodes[1] = heap->nodes[heap->size];
	heap->nodes[heap->size] = NULL;
	heap->size--;
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
		if (smallest == current) {
			break;
		}
		swap(heap, current, smallest);
		current = smallest;
	}
	return res;
}

__device__ bool heaps_empty(heap_t **heaps, int k) {
    for (int i = 0; i < k; i++) {
        if (heaps[i]->size != 0) return false;
    }
    return true;
}

__device__ int heaps_min(heap_t **heaps, int k) {
	float best_f = FLT_MAX;
	for (int i = 0; i < k; i++) {
		node_t *current_best = heaps[i]->nodes[1];
		if (current_best != NULL && current_best->fCost < best_f) {
			best_f = current_best->fCost;
		}
	}
	return best_f;
}
