#include <stdlib.h>
#include "closed_list.h"
#include "astar.h"

closed_list *closed_list_create(size_t capacity) {
    closed_list *list = malloc(sizeof(closed_list));
    list->capacity = capacity;
    list->nodes = calloc(capacity, sizeof(node *));
    return list;
}

void closed_list_destroy(closed_list *list) {
    free(list->nodes);
    free(list);
}

void closed_list_insert(closed_list *closed, node *n) {
    if (closed->nodes[n->id] == NULL || closed->nodes[n->id]->fCost > n->fCost) {
        closed->nodes[n->id] = n;
    }
}

node *closed_list_get(closed_list *closed, node *node) {
    return closed->nodes[node->id];
}