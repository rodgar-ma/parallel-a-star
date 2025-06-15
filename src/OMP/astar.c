#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include "astar.h"
#include "priority_list.h"
#include "MapUtils.h"
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
    #pragma omp parallel for
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
    #pragma omp parallel for
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
    int n_fallos = 0;

    omp_set_num_threads(k);

    double start = omp_get_wtime();

    #pragma omp parallel
    while (Q[0]->size > 0)
    {
        int tid = omp_get_thread_num();

        if (tid == 0) {
            list_clear(S);
        }

        #pragma omp barrier

        node *q = priority_list_extract(Q[k]);
            
        if (q->id == t_id) {
            m = q;
            #pragma omp critical
            {
                if (m == NULL || q->fCost < m->fCost) {
                    m = q;
                }
            }
            continue;
        }

        neighbors[0]->count = 0;
        source->get_neighbors(neighbors[0], q->id);

        for(int j = 0; j < neighbors[0]->count; j++) {
            int neighborId = neighbors[0]->nodeIds[j];
            double newCost = q->gCost + neighbors[0]->costs[j];
            if (!visited_list_contains(H, neighborId) || newCost < H->nodes[neighborId]->gCost) {
                if (visited_list_insert(H, neighborId, newCost, newCost + source->heuristic(neighborId, t_id), q)) {
                    priority_list_insert(Q[0], H->nodes[neighborId]);
                }
            }
        }
        // }

        // t1_2 = omp_get_wtime();
        // time_1 += t1_2 - t1_1;
        
        // printf("Lista S:\n");
        // for(int j = 0; j < S->capacity; j++) {
        //     if (S->ids[j] == -1) continue;
        //     Node neighbor = GetNodeById(MAP, S->ids[j]);
        //     printf("Nodo (%d, %d), gCost = %f.\n", neighbor->x, neighbor->y, S->gCosts[j]);
        // }
        // printf("\n");

        // if (m != NULL && m->fCost < priority_lists_min(Q, k)) {
        //     break;
        // }

        // t2_1 = omp_get_wtime();
        // #pragma omp parallel for schedule(static, 1)
        // for (int i = 0; i < S->capacity; i++) {
        //     if (S->ids[i] == -1) continue;
        //     // Node n = GetNodeById(MAP, S->ids[i]);
        //     if (!visited_list_contains(H, S->ids[i]) || S->gCosts[i] < H->nodes[S->ids[i]]->gCost) {
        //         if (visited_list_insert(H, S->ids[i], S->gCosts[i], S->gCosts[i] + source->heuristic(S->ids[i], t_id), S->parents[i])) {
        //             priority_list_insert(Q[(steps+i)%k], H->nodes[S->ids[i]]);
        //             // printf("Nodo (%d, %d), gCost = %f, fCost = %f, en lista de prioridad %d.\n", n->x, n->y, H->nodes[S->ids[i]]->gCost, H->nodes[S->ids[i]]->fCost, (steps+i)%k);
        //         } else {
        //             #pragma omp critical
        //             {
        //                 n_fallos++;
        //             }
                    // Node n = GetNodeById(MAP, S->ids[i]);
                    // printf("Nodo (%d, %d) llega tarde. old_fCost = %f, new_fCost = %f.\n", n->x, n->y, H->nodes[S->ids[i]]->gCost, S->gCosts[i]);
                // }
            // } // else {
                // printf("Nodo (%d, %d) no se actualiza. old_gCost = %f, new_gCost = %f.\n", n->x, n->y, H->nodes[S->ids[i]]->gCost, S->gCosts[i]);
            // }
        // }
        // t2_2 = omp_get_wtime();
        // time_2 += + t2_2 - t2_1;

        // printf("\n\n");
        steps++;
    }
    double end = omp_get_wtime();
    *cpu_time_used = (end - start) * 1000.0;
    // printf("Tiempo 1er bucle: %f\n", time_1 * 1000.0);
    // printf("Tiempo 2do bucle: %f\n", time_2);
    printf("%d iteraciones.\n", steps);
    printf("n_fallos = %d\n", n_fallos);
    printf("Nodo final: %d\n", m->id);
    path *path = retrace_path(m); // AQUÍ FALLA
    priority_lists_destroy(Q, k);
    visited_list_destroy(H);
    neighbors_lists_destroy(neighbors, k);
    // list_destroy(S);
    return path;
}