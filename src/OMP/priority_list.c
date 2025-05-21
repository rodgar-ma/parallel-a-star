#include <stdlib.h>
#include <float.h>
#include "priority_list.h"

static void swap(node **n1, node **n2);

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
    list->nodes = calloc(capacity, sizeof(node*));
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
        int i = list->size++;
        list->nodes[i] = n;

        // Heapify-up
        while (i > 0) {
            int parent = (i - 1) / 2;
            if (list->nodes[parent]->fCost <= list->nodes[i]->fCost) break;

            swap(&(list->nodes[i]), &(list->nodes[parent]));
            i = parent;
        }
}

node *priority_list_extract(priority_list *list) {
    node *res = list->nodes[0];
    list->nodes[0] = list->nodes[--list->size];

    int i = 0;
    while (1) {
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        int smallest = i;

        if (left < list->size && list->nodes[left]->fCost < list->nodes[smallest]->fCost) {
            smallest = left;
        }
        if (right < list->size && list->nodes[right]->fCost < list->nodes[smallest]->fCost) {
            smallest = right;
        }
        if (smallest == i) break;

        swap(&(list->nodes[i]), &(list->nodes[smallest]));
        i = smallest;
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
		node *current_best = lists[i]->nodes[0];
		if (current_best != NULL && current_best->fCost < best_f) {
			best_f = current_best->fCost;
		}
	}
	return best_f;
}

static void swap(node **n1,  node **n2) {
	node *tmp = *n1;
	*n1 = *n2;
	*n2 = tmp;
}