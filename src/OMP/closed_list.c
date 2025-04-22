#include <stdlib.h>
#include "closed_list.h"

closed_list *closed_list_create(size_t capacity) {
    closed_list *list = malloc(sizeof(closed_list));
    list->capacity = capacity;
    list->nodes = calloc(capacity, sizeof(node*));
    return list;
}

void closed_list_destroy(closed_list *list) {
    free(list->nodes);
    free(list);
}

int closed_list_contains(closed_list *list, node *n) {
    return list->nodes[n->id] != NULL;
}

void closed_list_insert(closed_list *list, node *n) {
    list->nodes[n->id] = n;
}

void closed_list_remove(closed_list *list, node *n) {
    list->nodes[n->id] = NULL;
}

int closed_list_is_better(closed_list *list, node *n) {
    return list->nodes[n->id]->gCost <= n->gCost;
}