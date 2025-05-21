#ifndef LIST_H
#define LIST_H

#include "astar.h"

typedef struct __list list;

struct __list {
    int length;
    int capacity;
    int *ids;
    double *gCosts;
    node **parents;
};

list **lists_create(int k, int capacity);

list *list_create(int capacity);

void lists_destroy(list **lists, int k);

void list_destroy(list *list);

void list_clear(list *list);

void list_insert(list *list, int id, double gCost, node *parent);

void list_remove(list *list, int index);

#endif