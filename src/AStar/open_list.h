#ifndef OPEN_LIST_H
#define OPEN_LIST_H

#include <stdlib.h>
#include "astar.h"

#define DEFAULT_QUEUE_SIZE 16

typedef struct __open_list {
    size_t capacity;
    size_t count;
    node **nodes;
} open_list;

open_list **open_lists_create(int k);

open_list *open_list_create();

void open_lists_destroy(open_list **lists, int k);

void open_list_destroy(open_list *list);

void open_list_insert_or_update(open_list *list, node *node);

node *open_list_extract(open_list *list);

int open_list_is_empty(open_list *list);

int open_lists_are_empty(open_list **lists, int k);

double open_list_get_min(open_list *list);

double open_lists_get_min(open_list **lists, int k);

#endif