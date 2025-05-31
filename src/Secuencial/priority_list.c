#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include "priority_list.h"

static void swap(priority_list *list, int id_1, int id_2) {
    int id_tmp = list->nodes[id_1];
    double prio_tmp = list->priorities[id_1];
    list->nodes[id_1] = list->nodes[id_2];
    list->priorities[id_1] = list->priorities[id_2];
    list->nodes[id_2] = id_tmp;
    list->priorities[id_2] = prio_tmp;
}

/* Crea una lista de prioridad */
priority_list *priority_list_create(int capacity) {
    priority_list *list = malloc(sizeof(priority_list));
    list->size = 0;
    list->capacity = capacity;
    list->nodes = calloc(capacity + 1, sizeof(int));
    list->priorities = calloc(capacity + 1, sizeof(double));
    return list;
}

/* Libera la memoria de la lista de prioridad */
void priority_list_destroy(priority_list *list) {
    free(list->nodes);
    free(list->priorities);
    free(list);
}

void priority_list_insert(priority_list *list, int n_id, double priority) {
    list->size++;
    if (list->size > list->capacity) {
        list->capacity *= 2;
        list->nodes = realloc(list->nodes, (list->capacity + 1) * sizeof(int));
        list->priorities = realloc(list->priorities, (list->capacity + 1) * sizeof(double));
    }
    list->nodes[list->size] = n_id;
    list->priorities[list->size] = priority;
    int current = list->size;
    while (current > 1 && list->priorities[current] < list->priorities[current / 2]) {
        swap(list, current, current / 2);
        current /= 2;
    }
}

int priority_list_extract(priority_list *list) {
    int res = list->nodes[1];
    list->nodes[1] = list->nodes[list->size];
    list->priorities[1] = list->priorities[list->size];
    list->size--;
    int current = 1;
    while(current < list->size) {
        int smallest = current;
        int child = 2 * current;
        if (child <= list->size && list->priorities[child] <= list->priorities[smallest]) {
            smallest = child;
        }
        child = 2 * current + 1;
        if (child <= list->size && list->priorities[child] <= list->priorities[smallest]) {
            smallest = child;
        }
        if (smallest == current) break;
        swap(list, current, smallest);;
        current = smallest;
    }
    return res;
}

int priority_lists_empty(priority_list **lists, int k) {
	for (int i = 0; i < k; i++) {
		if (lists[i]->size != 0) return 0;
	}
	return 1;
}
