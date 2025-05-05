#ifndef LIST_H
#define LIST_H

#include <stdlib.h>
#include "astar.h"

typedef struct __list list;

struct __list {
    size_t capacity;
    size_t count;
    node **nodes;
};

list *list_create();
void list_destroy(list *list);
void list_clear(list *list);
void list_insert(list *list, node *n);
void list_remove(list *list, size_t index);
node *list_get(list *list, size_t index);

#endif