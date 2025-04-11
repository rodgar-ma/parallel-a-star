#include <stdlib.h>
#include <float.h>
#include <omp.h>
#include "astar.h"
#include "list.h"
#include "open_list.h"
#include "closed_list.h"
#include "list.h"

#define HASH_SIZE 1024 * 1024
#define HASH_FUNS 2

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

neighbors_list **neighbors_lists_create(int k) {
    neighbors_list **lists = calloc(k, sizeof(neighbors_list *));
    for (int i = 0; i < k; i++) {
        lists[i] = neighbors_list_create();
    }
    return lists;
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

path *find_path_omp(AStarSource *source, id_t s_id, id_t t_id, int k) {
    open_list **Q = open_lists_create(k);
    closed_list *H = closed_list_create(source->max_size);
    list *S = list_create();
    neighbors_list **neighbors = neighbors_lists_create(k);
    open_list_insert_or_update(Q[0], node_create(s_id, 0, source->heuristic(s_id, t_id), NULL));

    omp_set_num_threads(k);

    node *m = NULL;
    int steps = 0;
    while (!open_lists_are_empty(Q, k))
    {
        list_clear(S);
        # pragma omp parallel for shared(k, Q, S, t_id, neighbors, source, m)
        for(int i = 0; i < k; i++) {
            if (open_list_is_empty(Q[i])) continue;
            node *q = open_list_extract(Q[i]);
            if (q->id == t_id) {
                if (m == NULL || q->gCost + source->heuristic(q->id, t_id) < m->gCost + source->heuristic(m->id, t_id)) {
                    m = q;
                }
                continue;
            }
            neighbors[i]->count = 0;
            source->get_neighbors(neighbors[i], q->id);
            for(int j = 0; j < neighbors[i]->count; j++) {
                #pragma omp critical
                {
                    list_insert(S, node_create(neighbors[i]->nodeIds[j], q->gCost + neighbors[i]->costs[j], DBL_MAX, q));
                }
            }
        }
        if (m != NULL && m->gCost + source->heuristic(m->id, t_id) < open_lists_get_min(Q, k)) {
            break;
        }
        # pragma omp parallel for
        for (size_t i = 0; i < S->count; i++) {
            node *s = list_get(S, i);
            if (closed_list_contains(H, s) && closed_list_is_better(H, s)) {
                list_remove(S, i);
            }
        }
        # pragma omp parallel for
        for (int i = 0; i < S->capacity; i++) {
            node *t1 = list_get(S, i);
            if (t1 != NULL) {
                t1->fCost = t1->gCost + source->heuristic(t1->id, t_id);
                open_list_insert_or_update(Q[(i + steps) % k], t1);
                closed_list_insert(H, t1);
            }
        }
        steps++;
    }
    path *path = reatrace_path(m);
    open_lists_destroy(Q, k);
    closed_list_destroy(H);
    list_destroy(S);
    return path;
}