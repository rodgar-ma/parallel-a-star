#include <stdlib.h>
#include <float.h>
#include <omp.h>
#include <stdio.h>
#include "astar.h"
#include "priority_list.h"
#include "closed_list.h"

node *node_create(astar_id_t id, double gCost, double fCost, node *parent) {
    node *n = malloc(sizeof(node));
    n->id = id;
    n->parent = parent;
    n->gCost = gCost;
    n->fCost = fCost;
    n->isOpen = 0;
    n->open_index = -1;
    return n;
}

neighbors_list *neighbors_list_create() {
    neighbors_list *list = malloc(sizeof(neighbors_list));
    list->capacity = DEFAULT_NIEGHBORS_LIST_CAPACITY;
    list->count = 0;
    list->costs = calloc(list->capacity, sizeof(double));
    list->nodeIds = calloc(list->capacity, sizeof(astar_id_t));
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

void neighbors_lists_destroy(neighbors_list** lists, int k) {
    for (int i = 0; i < k; i++) {
        neighbors_list_destroy(lists[i]);
    }
    free(lists);
}

void add_neighbor(neighbors_list *neighbors, astar_id_t n_id, double cost) {
    if (neighbors->count == neighbors->capacity) {
        neighbors->capacity *= 2;
        neighbors->nodeIds = realloc(neighbors->nodeIds, neighbors->capacity * sizeof(astar_id_t));
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

    p->nodeIds = calloc(p->count, sizeof(astar_id_t));
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

path *find_path_omp(AStarSource *source, astar_id_t s_id, astar_id_t t_id, int k) {
    omp_set_num_threads(k);

    priority_list **Q = priority_lists_create(k);
    closed_list *H = closed_list_create(source->max_size);
    node **S = calloc(MAX_S_SIZE, sizeof(node*));
    int S_count = 0;
    neighbors_list **neighbors = neighbors_lists_create(k);

    priority_list_insert_or_update(Q[0], node_create(s_id, 0, source->heuristic(s_id, t_id), NULL));

    node *m = NULL;
    int steps = 0;
    while (!priority_lists_are_empty(Q, k))
    {
        S_count = 0;
        #pragma omp parallel for num_threads(k) schedule(static,1)
        for(int i = 0; i < k; i++) {
            if (priority_list_is_empty(Q[i])) continue;
            node *q = priority_list_extract(Q[i]);
            if (q->id == t_id) {
                #pragma omp critical
                {
                    if (m == NULL || q->gCost + source->heuristic(q->id, t_id) < m->gCost + source->heuristic(m->id, t_id)) {
                        m = q;
                    }
                }
                continue;
            }
            neighbors[i]->count = 0;
            source->get_neighbors(neighbors[i], q->id);
            for(int j = 0; j < neighbors[i]->count; j++) {
                int idx;
                #pragma omp atomic capture
                idx = S_count++;
                if (idx < MAX_S_SIZE) {
                    S[idx] = node_create(neighbors[i]->nodeIds[j], q->gCost + neighbors[i]->costs[j], DBL_MAX, q);
                } else {
                    printf("Nodo fuera de rango en la lista S");
                }
            }
        }
        if (m != NULL && m->gCost + source->heuristic(m->id, t_id) < priority_lists_get_min(Q, k)) {
            break;
        }
        
        #pragma omp parallel for num_threads(k) schedule(static,1)
        for (size_t i = 0; i < S_count; i++) {
            node *s = S[i];
            omp_set_lock(&H->locks[s->id]);
            if (closed_list_contains(H, s) && closed_list_is_better(H, s)) {
                S[i] = NULL;
            }
            omp_unset_lock(&H->locks[s->id]);
        }
        #pragma omp parallel for num_threads(k) schedule(static,1)
        for (int i = 0; i < S_count; i++) {
            node *t1 = S[i];
            if (t1 != NULL) {
                t1->fCost = t1->gCost + source->heuristic(t1->id, t_id);
                priority_list_insert_or_update(Q[(i + steps) % k], t1);
                closed_list_insert(H, t1);
            }
        }
        steps++;
    }
    path *path = reatrace_path(m);
    priority_lists_destroy(Q, k);
    closed_list_destroy(H);
    neighbors_lists_destroy(neighbors, k);
    free(S);
    return path;
}