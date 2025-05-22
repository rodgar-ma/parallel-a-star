#include <stdlib.h>
#include <omp.h>
#include <assert.h>
#include "list.h"


list *list_create(int capacity) {
	list *l = malloc(sizeof(list));
    l->capacity = capacity;
    l->ids = calloc(capacity, sizeof(int));
	l->gCosts = calloc(capacity, sizeof(double));
	l->parents = calloc(capacity, sizeof(node*));
    return l;
}

void list_destroy(list *list) {
	free(list->ids);
	free(list->gCosts);
	free(list->parents);
    free(list);
}

void list_clear(list *list) {
	#pragma omp parallel for
	for(int i = 0; i < list->capacity; i++) {
		list->ids[i] = -1;
	}
}

void list_insert(list *list, int id, neighbors_list *neighbors, node *parent) {
	int index = MAX_NODE_EXPAND * id;
	for(int i = 0; i < neighbors->count; i++) {
		list->ids[index + i] = neighbors->nodeIds[i];
		list->gCosts[index + i] = parent->gCost + neighbors->costs[i];
		list->parents[index + i] = parent;
	}
}
