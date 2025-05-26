#ifndef VISITED_LIST_H
#define VISITED_LIST_H

#include <omp.h>
#include "astar.h"
#include "list.h"

typedef struct __visited_list visited_list;

struct __visited_list {
    int capacity;
    node** nodes;
    int use_locks;
    omp_lock_t* locks;
};

visited_list *visited_list_create(int capacity, int locks);

void visited_list_destroy(visited_list *H);

int visited_list_contains(visited_list *H, int node_id);

void visited_list_insert(visited_list *H, int id, double gCost, double fCost, node *parent);

int visited_list_is_better(visited_list *H, int id, double fCost);


#endif