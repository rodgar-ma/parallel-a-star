#ifndef HASH_TABLE_H
#define HASH_TABLE_H

#include <stdlib.h>
#include "astar.h"

typedef struct __closed_list {
    size_t capacity;
    node **nodes;
} closed_list;

closed_list *closed_list_create(size_t capacity);
void closed_list_destroy(closed_list *list);
void closed_list_insert(closed_list *closed, node *node);
node *closed_list_get(closed_list *closed, void *node);
void closed_list_remove(closed_list *closed, size_t index);


#endif