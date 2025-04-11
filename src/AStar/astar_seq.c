#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include "astar.h"
#include "open_list.h"
#include "closed_list.h"

node *node_create(id_t id, double gCost, double fCost, node *parent) {
    node *n = malloc(sizeof(node));
    n->id = id;
    n->parent = parent;
    n->gCost = gCost;
    n->fCost = fCost;
    return n;
}

neighbors_list *neighbors_list_create() {
    neighbors_list *list = malloc(sizeof(neighbors_list));
    list->capacity = DEFAULT_NIEGHBORS_LIST_CAPACITY;
    list->count = 0;
    list->costs = calloc(list->capacity, sizeof(double));
    list->nodeIds = calloc(list->capacity, sizeof(id_t));
    return list;
}

void neighbors_list_destroy(neighbors_list *list) {
    free(list->nodeIds);
    free(list->costs);
    free(list);
}

void add_neighbor(neighbors_list *neighbors, id_t n_id, double cost) {
    if (neighbors->count == neighbors->capacity) {
        neighbors->capacity *= 2;
        neighbors->nodeIds = realloc(neighbors->nodeIds, neighbors->capacity * sizeof(id_t));
        neighbors->costs = realloc(neighbors->costs, neighbors->capacity * sizeof(double));
    }
    neighbors->nodeIds[neighbors->count] = n_id;
    neighbors->costs[neighbors->count] = cost;
    neighbors->count++;
}

path *reatrace_path(node *target) {
    path *p = malloc(sizeof(path));
    p->count = 0;
    p->cost = target->fCost;

    node *current = target;
    while (current) {
        p->count++;
        current = current->parent;
    }

    p->nodeIds = calloc(p->count, sizeof(id_t));
    current = target;
    for (int i = 0; i < p->count; i++) {
        p->nodeIds[p->count-i-1] = target->id;
        current = current->parent;
    }
    return p;
}

void path_destroy(path *p) {
    free(p->nodeIds);
    free(p);
}

path *find_path_sequential(AStarSource *source, id_t s_id, id_t t_id) {
    open_list *open = open_list_create();
    closed_list *closed = closed_list_create(source->max_size);
    neighbors_list *neighbors = neighbors_list_create();

    node *current = node_create(s_id, 0, source->heuristic(s_id, t_id), NULL);
    open_list_insert_or_update(open, current);

    while(!open_list_is_empty(open)) {
        current = open_list_extract(open);
        closed_list_insert(closed, current);
        
        if (current->id == t_id) {
            break;
        }

        neighbors->count = 0;
        source->get_neighbors(neighbors, current->id);
        for (size_t i = 0; i < neighbors->count; i++) {
            double newCost = current->gCost + neighbors->costs[i];
            node *neighbor = node_create(neighbors->nodeIds[i], newCost, newCost + source->heuristic(neighbors->nodeIds[i], t_id), current);
            if (!closed_list_contains(closed, neighbor) || !closed_list_is_better(closed, neighbor)) {
                closed_list_remove(closed, neighbor);
                open_list_insert_or_update(open, neighbor);
            }
        }
    }
    path *path = reatrace_path(current);
    open_list_destroy(open);
    closed_list_destroy(closed);
    neighbors_list_destroy(neighbors);
    return path;
}
