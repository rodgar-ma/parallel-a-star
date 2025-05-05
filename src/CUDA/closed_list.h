#ifndef CLOSED_LIST_H
#define CLOSED_LIST_H

#include <stdlib.h>
#include <omp.h>
#include "astar.h"

typedef struct __closed_list closed_list;

struct __closed_list {
    size_t capacity;
    node** nodes;
    omp_lock_t* locks;
};

closed_list *closed_list_create(size_t capacity);

void closed_list_destroy(closed_list *list);

int closed_list_contains(closed_list *list, node *n);

void closed_list_insert(closed_list *list, node *n);

void closed_list_remove(closed_list *list, node *n);

int closed_list_is_better(closed_list *list, node *n);

#endif