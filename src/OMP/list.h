#ifndef LIST_H
#define LIST_H

#include "astar.h"

typedef struct __list list;

struct __list {
    int capacity;
    int *ids;
    double *gCosts;
    node **parents;
};


list *list_create(int capacity);

void list_destroy(list *list);

void list_clear(list *list);

#endif