#include <stdlib.h>
#include "list.h"
#include "astar.h"

list *list_create() {
    list *list = malloc(sizeof(struct __list));
    list->capacity = 0;
    list->count = 0;
    list->nodes = NULL;
    return list;
}

void list_destroy(list *list) {
    free(list->nodes);
    free(list);
}

void list_clear(list *list) {
    list->count = 0;
}

void list_insert(list *list, node *n) {
    if (list->count == list->capacity) {
        list->capacity = 1 + (2 * list->capacity);
        list->nodes = realloc(list->nodes, list->capacity * sizeof(node));
    }
    list->nodes[list->count++] = n;
}

void list_remove(list *list, size_t index) {
    if (index < list->capacity) list->nodes[index] = NULL;
}

node *list_get(list *list, size_t index) {
    if (index < list->capacity) return list->nodes[index];
    return NULL;
}