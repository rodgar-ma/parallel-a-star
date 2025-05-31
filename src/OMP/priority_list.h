#ifndef PRIORITY_LIST_H
#define PRIORITY_LIST_H

#include "astar.h"

typedef struct __priority_item {
    node *node;
    double priority;
} priority_item;

typedef struct __priority_list {
    int size;
    priority_item *items;
} priority_list;

priority_list **priority_lists_create(int k);

priority_list *priority_list_create(int capacity);

void priority_lists_destroy(priority_list **lists, int k);

void priority_list_destroy(priority_list *list);

void priority_list_insert(priority_list *list, node *node);

node *priority_list_extract(priority_list *list);

int priority_lists_empty(priority_list **lists, int k);

double priority_lists_min(priority_list **lists, int k);

#endif