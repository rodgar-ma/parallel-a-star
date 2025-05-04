#ifndef PRIORITY_LIST_H
#define PRIORITY_LIST_H

#include <stdlib.h>
#include "astar.h"

#define DEFAULT_QUEUE_SIZE 16

typedef struct __priority_list {
    size_t capacity;
    size_t count;
    node **nodes;
} priority_list;

priority_list *priority_list_create();

void priority_list_destroy(priority_list *list);

void priority_list_insert_or_update(priority_list *list, node *n);

node *priority_list_extract(priority_list *list);

int priority_list_is_empty(priority_list *list);

double priority_list_get_min(priority_list *list);

#endif