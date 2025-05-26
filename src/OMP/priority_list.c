#include <stdlib.h>
#include <float.h>
#include "priority_list.h"

#define MAX_QUEUE_SIZE 1000000

// static void swap(node **n1, node **n2);

// static void swap(priority_list *list, int i, int j) {
//     node *tmp_node = list->nodes[i];
//     float tmp_prio = list->priorities[i];
//     list->nodes[i] = list->nodes[j];
//     list->priorities[i] = list->priorities[j];
//     list->nodes[j] = tmp_node;
//     list->priorities[j] = tmp_prio;
// }

static void swap(node **n1, node **n2) {
    node *tmp = *n1;
    *n1 = *n2;
    *n2 = tmp;
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
    list->nodes = calloc(capacity + 1, sizeof(node*));
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
    free(list->nodes);
    free(list);
}

void priority_list_insert(priority_list *list, node *n) {
    list->size++;
    list->nodes[list->size] = n;
    int current = list->size;
    while (current > 1 && list->nodes[current]->fCost < list->nodes[current / 2]->fCost) {
        swap(&(list->nodes[current]), &(list->nodes[current / 2]));
        current /= 2;
    }
}

node *priority_list_extract(priority_list *list) {
    node *res = list->nodes[1];
    list->nodes[1] = list->nodes[list->size];
    list->nodes[list->size] = NULL;
    list->size--;
    int current = 1;
    while(current < list->size) {
        int smallest = current;
        int child = 2 * current;
        if (child <= list->size && list->nodes[child]->fCost <= list->nodes[smallest]->fCost) {
            smallest = child;
        }
        child = 2 * current + 1;
        if (child <= list->size && list->nodes[child]->fCost <= list->nodes[smallest]->fCost) {
            smallest = child;
        }
        if (smallest == current) break;
        swap(&(list->nodes[current]), &(list->nodes[smallest]));
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
		node *current_best = lists[i]->nodes[1];
		if (current_best != NULL && current_best->fCost < best_f) {
			best_f = current_best->fCost;
		}
	}
	return best_f;
}