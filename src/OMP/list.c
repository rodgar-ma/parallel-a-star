#include <stdlib.h>
#include <omp.h>
#include <assert.h>
#include "list.h"

list **lists_create(int k, int capacity) {
	list **lists = calloc(k, sizeof(list*));
	for (int i = 0; i < k; i++) {
		lists[i] = list_create(capacity);
	}
	return lists;
}

list *list_create(int capacity) {
	list *l = malloc(sizeof(list));
    l->length = 0;
    l->capacity = capacity;
    l->ids = calloc(capacity, sizeof(int));
	l->gCosts = calloc(capacity, sizeof(double));
	l->parents = calloc(capacity, sizeof(node*));
    return l;
}

void lists_destroy(list **lists, int k) {
	for (int i = 0; i < k; i++) {
		list_destroy(lists[i]);
	}
	free(lists);
}

void list_destroy(list *list) {
	free(list->ids);
	free(list->gCosts);
	free(list->parents);
    free(list);
}

void list_clear(list *list) {
	list->length = 0;
}

void list_insert(list *list, int id, double gCost, node *parent) {
	int index;
    #pragma omp atomic capture
	index = list->length++;
    assert(index < list->capacity);
	list->ids[index] = id;
	list->gCosts[index] = gCost;
	list->parents[index] = parent;
}

void list_remove(list *list, int index) {
	assert(list->length < list->capacity);
	list->ids[index] = -1;
}
