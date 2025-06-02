#ifndef PRIORITY_LIST_H
#define PRIORITY_LIST_H

#include "astar.h"

typedef struct __priority_item {
    int node;
    double priority;
} priority_item;

typedef struct __priority_list {
    int size;
    int capacity;
    int *nodes;
    double *priorities;
} priority_list;

priority_list *priority_list_create();

void priority_list_destroy(priority_list *list);

void priority_list_insert(priority_list *list, int n_id, double priority);

int priority_list_extract(priority_list *list);

int priority_lists_empty(priority_list **lists, int k);

double priority_lists_min(priority_list **lists, int k);

#endif