#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include "astar.h"
#include "open_list.h"
#include "closed_list.h"

node *node_create(unsigned long *id, double gCost, double fCost, node *parent) {
    node *node = malloc(sizeof(struct __node));
    node->id = id;
    node->parent = parent;
    node->gCost = gCost;
    node->fCost = fCost;
    return node;
}

neighbors_list *neighbors_list_create() {
    neighbors_list *list = malloc(sizeof(neighbors_list));
    list->capacity = 0;
    list->count = 0;
    list->costs = NULL;
    list->nodes = NULL;
    return list;
}

void neighbors_list_destroy(neighbors_list *neighbors) {
    free(neighbors->costs);
    free(neighbors);
}

void add_neighbor(neighbors_list *neighbors, unsigned long *node, double cost) {
    if (neighbors->count == neighbors->capacity) {
        neighbors->capacity = 1 + (2 * neighbors->capacity);
        neighbors->costs = realloc(neighbors->costs, neighbors->capacity * sizeof(double));
        neighbors->nodes = realloc(neighbors->nodes, neighbors->capacity * sizeof(unsigned long *));
    }
    neighbors->costs[neighbors->count] = cost;
    neighbors->nodes[neighbors->count] = node;
    neighbors->count++;
}

path *path_retrace(node *goal) {
    path *path = malloc(sizeof(struct __path));
    path->cost = goal->gCost;
    path->count = 0;

    node *current = goal;
    while (current) {
        path->count++;
        current = current->parent;
    }

    current = goal;
    path->nodes = calloc(path->count, sizeof(unsigned long *));
    for (size_t i = 0; i < path->count; i++) {
        path->nodes[path->count-i-1] = current->id;
        current = current->parent;
    }
    return path;
}

void path_destroy(path *path) {
    free(path->nodes);
    free(path);
}

path *find_path_sequential(AStarSource *source, unsigned long start, unsigned long goal) {
    open_list *open = open_list_create();
    closed_list *closed = closed_list_create(source->max_size);
    neighbors_list *neighbors = neighbors_list_create();

    node *current = node_create(start, 0, source->heuristic(start, goal), NULL);

    open_list_insert(open, current);

    while(!open_list_is_empty(open)) {
        current = open_list_extract(open);
        closed_list_insert(closed, current);
        
        if (current->id == goal) {
            break;
        }
        
        neighbors->count = 0;
        source->get_neighbors(current->id, neighbors);
        for (size_t i = 0; i < neighbors->count; i++) {
            node *neighbor;
            double newCost = current->gCost + neighbors->costs[i];
            if ((neighbor = closed_list_get(closed, neighbors->nodes[i])) == NULL) {
                neighbor = node_create(neighbors->nodes[i], newCost, newCost + source->heuristic(neighbor->id, goal), current);
                open_list_insert(open, neighbor);
            } else if (newCost < neighbor->gCost) {
                neighbor->gCost = newCost;
                neighbor->fCost = newCost + source->heuristic(neighbor->id, goal);
                neighbor->parent = current;
                closed_list_remove(closed, neighbor);
                open_list_insert(open, neighbor);
            }
        }
    }
    path *path = path_retrace(current);
    closed_list_destroy(closed);
    open_list_destroy(open);
    neighbors_list_destroy(neighbors);
    return path;
}
