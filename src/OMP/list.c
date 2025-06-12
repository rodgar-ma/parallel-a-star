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
	for(int i = 0; i < list->capacity; i++) {
		list->ids[i] = -1;
	}
}

