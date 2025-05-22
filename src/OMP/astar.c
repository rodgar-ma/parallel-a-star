#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <omp.h>
#include <time.h>
#include "astar.h"
#include "priority_list.h"
#include "visited_list.h"
#include "list.h"

node *node_create(int id, double gCost, double fCost, node *parent) {
    node *n = malloc(sizeof(node));
    n->id = id;
    n->gCost = gCost;
    n->fCost = fCost;
    n->parent = parent;
    return n;
}

neighbors_list *neighbors_list_create() {
    neighbors_list *list = malloc(sizeof(neighbors_list));
    list->capacity = MAX_NODE_EXPAND;
    list->count = 0;
    list->costs = calloc(list->capacity, sizeof(double));
    list->nodeIds = calloc(list->capacity, sizeof(int));
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

void add_neighbor(neighbors_list *neighbors, int n_id, double cost) {
    if (neighbors->count == neighbors->capacity) {
        return;
    }
    neighbors->nodeIds[neighbors->count] = n_id;
    neighbors->costs[neighbors->count] = cost;
    neighbors->count++;
}

path *retrace_path(node *target) {
    path *p = malloc(sizeof(path));
    p->count = 0;
    p->cost = target->fCost;

    node *current = target;
    while (current) {
        p->count++;
        current = current->parent;
    }

    p->nodeIds = calloc(p->count, sizeof(int));
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



/**********************************************************************************************************/
/*                                              A* Algorithm                                              */
/**********************************************************************************************************/

path *find_path_omp(AStarSource *source, int s_id, int t_id, int k, double *cpu_time_used) {

    priority_list **Q = priority_lists_create(k);
    visited_list *H = visited_list_create(source->max_size);
    list *S = list_create(k * MAX_NODE_EXPAND);
    neighbors_list **neighbors = neighbors_lists_create(k);

    H->nodes[s_id] = node_create(s_id, 0, source->heuristic(s_id, t_id), NULL);
    priority_list_insert(Q[0], H->nodes[s_id]);

    node *m = NULL;
    int steps = 0;

    clock_t start_time, end_time;
    start_time = clock();
    while (!priority_lists_empty(Q, k))
    {
        list_clear(S);
        #pragma omp parallel for shared(Q, S, neighbors, H, t_id, k, m, source) schedule(static, 1)
        for(int i = 0; i < k; i++) {
            if (Q[i]->size == 0) continue;
            node *q = priority_list_extract(Q[i]);
            if (q->id == t_id) {
                if (m == NULL || q->gCost + source->heuristic(q->id, t_id) < m->gCost + source->heuristic(m->id, t_id)) {
                    m = q;
                }
                continue;
            }
            neighbors[i]->count = 0;
            source->get_neighbors(neighbors[i], q->id);
            list_insert(S, i, neighbors[i], q);
        }

        if (m != NULL && m->gCost + source->heuristic(m->id, t_id) < priority_lists_min(Q, k)) {
            break;
        }
        #pragma omp parallel for shared(Q, S, neighbors, H, t_id, k, steps, m, source)
        for (int i = 0; i < S->capacity; i++) {
            if (S->ids[i] == -1) continue;
            if (visited_list_contains(H, S->ids[i]) && visited_list_is_better(H, S, i)) {
                continue;
            } else {
                visited_list_insert(H, S, i, source->heuristic(S->ids[i], t_id));
                priority_list_insert(Q[(steps+i)%k], H->nodes[S->ids[i]]);
            }
        }
        steps++;
    }
    end_time = clock();
    *cpu_time_used = (double) (end_time - start_time) / CLOCKS_PER_SEC;
    path *path = retrace_path(m);
    priority_lists_destroy(Q, k);
    visited_list_destroy(H);
    neighbors_lists_destroy(neighbors, k);
    list_destroy(S);
    return path;
}