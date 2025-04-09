#include <stdlib.h>
#include "astar.h"
#include "list.h"
#include "priority_queue.h"

#define HASH_SIZE 1024 * 1024
#define HASH_FUNS 2

node *node_create(void *n, double gCost, double fCost, node *parent) {
    node *new = malloc(sizeof(node));
    new->node = n;
    new->gCost = gCost;
    new->fCost = fCost;
    new->parent = parent;
    return new;
}

neighbors_list *neighbors_list_create() {
    neighbors_list *list = malloc(sizeof(neighbors_list));
    list->capacity = 0;
    list->count = 0;
    list->costs = NULL;
    list->elements = NULL;
    return list;
}

void add_neighbor(neighbors_list *neighbors, void *n, double cost) {
    if (neighbors->count == neighbors->capacity) {
        neighbors->capacity = 1 + (2 * neighbors->capacity);
        neighbors->costs = realloc(neighbors->costs, neighbors->capacity * sizeof(double));
        neighbors->elements = realloc(neighbors->elements, neighbors->capacity * sizeof(void*));
    }
    neighbors->elements[neighbors->count] = n;
    neighbors->costs[neighbors->count] = cost;
    neighbors->count++;
}

unsigned int jenkins_hash(int j, char *str) {
	unsigned long hash = (j * 10000007);
    char c = *str++;
	while (c) {
		hash += c;
		hash += hash << 10;
		hash ^= hash >> 6;
        c = *str++;
	}
	hash += hash << 3;
	hash ^= hash >> 11;
	hash += hash << 15;
	return hash;
}

path *retrace_path(node *goal) {
    path *path = malloc(sizeof(path));
    path->cost = goal->gCost;
    path->count = 0;
    node *current = goal;
    while (current) {
        path->count++;
        current = current->parent;
    }

    path->nodes = calloc(path->count, sizeof(void*));
    current = goal;
    for (int i = 0; i < path->count; i++) {
        path->nodes[path->count - (i+1)] = current;
        current = current->parent;
    }
    return path;
}

path *find_path(AStarSource source, void *start, void *target, int k) {
    node **H = calloc(HASH_SIZE, sizeof(node*));
    priority_queue **Q = priority_queues_create(k);
    list *S = list_create();
    priority_queue_insert(Q[0], node_create(start, 0, source.heuristic(start, target), NULL));

    node *m = NULL;
    int steps = 0;
    while (!priority_queues_are_empty(Q, k))
    {
        list_clear(S);
        neighbors_list *neighbors = neighbors_list_create();
        for(int i = 0; i < k; i++) {
            if (priority_queue_is_empty(Q[i])) continue;
            node *q = priority_queue_extract(Q[i]);
            if (q->node == target) {
                if (m == NULL || q->gCost + source.heuristic(q->node, target) < m->gCost + source.heuristic(m->node, target)) {
                    m = q;
                }
                continue;
            }
            source.get_neighbors(q->node, neighbors);
            for(int i = 0; i < neighbors->count; i++) {
                list_insert(S, node_create(neighbors->elements[i], q->gCost + neighbors->costs[i], -1, q));
            }
        }
        if (m != NULL && m->gCost + source.heuristic(m, target) < priority_queues_get_min(Q, k)) {
            break;
        }
        for (size_t i = 0; i < S->count; i++) {
            int z = 0;
            node *t1 = list_get(S, i);
            for (int j = 0; j < HASH_FUNS; j++) {
                node *visited = H[jenkins_hash(j, t1->node) % HASH_SIZE];
                if (visited == NULL || t1->node != visited->node) {
                    z = j;
                    break;
                }
            }
            int index = jenkins_hash(z, t1->node) % HASH_SIZE;
            t1 = H[index];
            if (t1 != NULL && t1->node == list_get(S, i)->node && list_get(S,i)->gCost + source.heuristic(list_get(S,i)->node, target) > t1->gCost + source.heuristic(t1->node, target)) {
                list_remove(S, i);
                continue;
            }
            t1 = list_get(S, i);
            for (int j = 0; j < HASH_FUNS; j++) {
                if (j != z) {
                    node *visited = H[jenkins_hash(j, t1->node) % HASH_SIZE];
                    if (visited != NULL && t1->node == visited->node && list_get(S, i)->gCost + source.heuristic(list_get(S, i)->node, target) > t1->gCost + source.heuristic(t1->node, target)) {
                        list_remove(S, i);
                        break;
                    }
                }
            }
        }
        for (int i = 0; i < S->capacity; i++) {
            node *t1 = list_get(S, i);
            if (t1 != NULL) {
                t1->fCost = t1->gCost + source.heuristic(t1->node, target);
                priority_queue_insert(Q[(i + steps) % k], t1);
            }
        }
        steps++;
    }
    return retrace_path(m);
}

void path_destroy(path *path) {
    free(path->nodes);
    free(path);
}