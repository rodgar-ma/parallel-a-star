#include <stdlib.h>
#include <limits.h>
#include <cuda.h>
#include <assert.h>
#include "list.h"
#include "cuda_utils.h"

list_t **lists_create(int lists, int capacity) {
    list_t **lists_cpu = (list_t**) malloc(lists * sizeof(list_t*));
    list_t **lists_gpu = NULL;
    for (int i = 0; i < lists; i++)
    {
        lists_cpu[i] = list_create(capacity);
    }
    HANDLE_RESULT(cudaMalloc(&lists_gpu, lists * sizeof(list_t*)));
    HANDLE_RESULT(cudaMemcpy(lists_gpu, lists_cpu, lists * sizeof(list_t*), cudaMemcpyDefault));
    return lists_gpu;
}

list_t *list_create(int capacity) {
    list_t list_cpu;
    list_t *list_gpu;
    list_cpu.size = 0;
    list_cpu.capacity = capacity;
    HANDLE_RESULT(cudaMalloc(&(list_cpu.nodes), (capacity + 1) * sizeof(node_t*)));
    HANDLE_RESULT(cudaMalloc(&list_gpu, sizeof(list_t)));
    HANDLE_RESULT(cudaMemcpy(list_gpu, &list_cpu, sizeof(list_t), cudaMemcpyDefault));
	return list_gpu;
}

void lists_destroy(list_t **lists_gpu, int lists) {
    list_t **lists_cpu = (list_t**)malloc(lists * sizeof(list_t*));
    HANDLE_RESULT(cudaMemcpy(lists_cpu, lists_gpu, lists * sizeof(list_t*), cudaMemcpyDefault));
	for (int i = 0; i < lists; i++) {
		list_destroy(lists_cpu[i]);
	}
	HANDLE_RESULT(cudaFree(lists_gpu));
	free(lists_cpu);
}

void list_destroy(list_t *list_gpu) {
	list_t list_cpu;
	HANDLE_RESULT(cudaMemcpy(&list_cpu, list_gpu, sizeof(list_t), cudaMemcpyDefault));
	HANDLE_RESULT(cudaFree(list_cpu.nodes));
	HANDLE_RESULT(cudaFree(list_gpu));
}

__device__ void list_clear(list_t *list) {
	list->size = 0;
}

__device__ void list_insert(list_t *list, node_t *node) {
	int index = atomicAdd(&(list->size), 1);
	assert(index < list->capacity);
	list->nodes[index] = node;
}


__device__ void list_remove(list_t *list, int index) {
	assert(list->size < list->capacity);
	list->nodes[index] = NULL;
}

__device__ node_t *list_get(list_t *list, int index) {
	assert(index < list->size);
	return list->nodes[index];
}