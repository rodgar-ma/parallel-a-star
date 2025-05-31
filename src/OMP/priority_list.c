#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include "priority_list.h"

#define MAX_QUEUE_SIZE 1000000

static void swap(priority_item *n1, priority_item *n2) {
    priority_item tmp = *n1;
    n1->node = n2->node;
    n1->priority = n2->priority;
    n2->node = tmp.node;
    n2->priority = tmp.priority;
}

/* Crea un conjunto de listas de prioridad */
priority_list **priority_lists_create(int k) {
    priority_list **lists = calloc(k, sizeof(priority_list*));
    for(int i = 0; i < k; i++) {
        lists[i] = priority_list_create(MAX_QUEUE_SIZE);
    }
    return lists;
}

/* Crea una lista de prioridad */
priority_list *priority_list_create(int capacity) {
    priority_list *list = malloc(sizeof(priority_list));
    list->size = 0;
    list->items = calloc(capacity + 1, sizeof(priority_item));
    return list;
}

/* Libera la memoria de un conjunto de listas de prioridad */
void priority_lists_destroy(priority_list **lists, int k) {
    for (int i = 0; i < k; i++) {
        priority_list_destroy(lists[i]);
    }
    free(lists);
}

/* Libera la memoria de la lista de prioridad */
void priority_list_destroy(priority_list *list) {
    free(list->items);
    free(list);
}

void priority_list_insert(priority_list *list, node *n) {
    list->size++;
    list->items[list->size].node = n;
    list->items[list->size].priority = n->fCost;
    int current = list->size;
    while (current > 1 && list->items[current].priority < list->items[current / 2].priority) {
        swap(&(list->items[current]), &(list->items[current / 2]));
        current /= 2;
    }
}

node *priority_list_extract(priority_list *list) {
    node *res = list->items[1].node;
    list->items[1] = list->items[list->size--];
    int current = 1;
    while(current < list->size) {
        int smallest = current;
        int child = 2 * current;
        if (child <= list->size && list->items[child].priority <= list->items[smallest].priority) {
            smallest = child;
        }
        child = 2 * current + 1;
        if (child <= list->size && list->items[child].priority <= list->items[smallest].priority) {
            smallest = child;
        }
        if (smallest == current) break;
        swap(&(list->items[current]), &(list->items[smallest]));
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

double priority_lists_min(priority_list **lists, int k) {
	double best_f = DBL_MAX;
	for (int i = 0; i < k; i++) {
		priority_item current_best = lists[i]->items[1];
		if (current_best.node != NULL && current_best.priority < best_f) {
			best_f = current_best.priority;
		}
	}
	return best_f;
}