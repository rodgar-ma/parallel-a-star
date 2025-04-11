#include <stdlib.h>
#include <float.h>
#include "astar.h"
#include "open_list.h"

open_list **open_lists_create(int k) {
    open_list **lists = calloc(k, sizeof(open_list*));
    for (int i = 0; i < k; i++) {
        lists[i] = open_list_create();
    }
    return lists;
}

open_list *open_list_create() {
    open_list *list = malloc(sizeof(open_list));
    list->capacity = INITIAL_CAPACITY;
    list->count = 0;
    list->nodes = NULL;
    return list;
}

void open_list_destroy(open_list *list) {
    free(list->nodes);
    free(list);
}

static void swap(node **a, node **b) {
    node *temp = *a;
    *a = *b;
    *b = temp;
}

void open_list_insert(open_list *list, node *n) {
    if (list->count == list->capacity) {
        list->capacity = 1 + (2 * list->capacity);
        list->nodes = realloc(list->nodes, list->capacity * sizeof(node*));
    }

    size_t i = list->count++;
    list->nodes[i] = n;

    while (i > 0 && list->nodes[(i-1)/2]->fCost > list->nodes[i]->fCost) {
        swap(&list->nodes[i], &list->nodes[(i-1)/2]);
        i = (i - 1) / 2;
    }
}

node *open_list_extract(open_list *list) {
    node *minNode = list->nodes[0];
    list->nodes[0] = list->nodes[--list->count];

    size_t i = 0;
    while (2 * i + 1 < list->count) {
        size_t left = 2 * i + 1;
        size_t right = 2 * i + 2;
        size_t smallest = left;

        if (right < list->count && list->nodes[right]->fCost < list->nodes[left]->fCost) {
            smallest = right;
        }

        if (list->nodes[i]->fCost <= list->nodes[smallest]->fCost) break;

        swap(&list->nodes[i], &list->nodes[smallest]);
        i = smallest;
    }
    return minNode;
}

int open_list_is_empty(open_list *list) {
    return list->count == 0;
}

int open_lists_are_empty(open_list **lists) {
    size_t index = 0;
    while (lists[index]) {
        if (lists[index]->count > 0) return 1;
        index++;
    }
    return 0;
}

double open_list_get_min(open_list *list) {
    return list->nodes[0]->fCost;
}

double open_lists_get_min(open_list **lists, int k) {
    double min_f = DBL_MAX;
    for (int i = 0; i < k; i++) {
        node *current_best = lists[i]->nodes[0];
        if (current_best != NULL && current_best->fCost < min_f) min_f = current_best->fCost;
    }
    return min_f;
}