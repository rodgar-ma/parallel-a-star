#include <stdio.h>
#include <omp.h>
#include <float.h>
#include "astar.h"
#include "heap.h"

node_t *node_create(int id, float gCost, float fCost, int parent) {
    node_t *n = malloc(sizeof(node_t));
    n->id = id;
    n->parent = parent;
    n->gCost = gCost;
    n->fCost = fCost;
    n->is_open = 0;
    n->open_index = -1;
    return n;
}

neighbors_list *neighbors_list_create() {
    neighbors_list *list = malloc(sizeof(neighbors_list));
    list->capacity = INIT_NEIGHBORS_LIST_CAPACITY;
    list->count = 0;
    list->costs = malloc(list->capacity * sizeof(float));
    list->nodeIds = malloc(list->capacity * sizeof(int));
    return list;
}

void neighbors_list_destroy(neighbors_list *list) {
    free(list->nodeIds);
    free(list->costs);
    free(list);
}

void add_neighbor(neighbors_list *neighbors, int n_id, float cost) {
    if (neighbors->count == neighbors->capacity) {
        neighbors->capacity *= 2;
        neighbors->nodeIds = realloc(neighbors->nodeIds, neighbors->capacity * sizeof(int));
        neighbors->costs = realloc(neighbors->costs, neighbors->capacity * sizeof(float));
    }
    neighbors->nodeIds[neighbors->count] = n_id;
    neighbors->costs[neighbors->count] = cost;
    neighbors->count++;
}

path *retrace_path(node_t ** closed, int target) {
    path *p = malloc(sizeof(path));
    p->count = 0;
    p->cost = closed[target]->gCost;

    int current = target;
    while (closed[current]->parent != -1) {
        p->count++;
        current = closed[current]->parent;
    }

    p->nodeIds = malloc(p->count * sizeof(int));
    current = target;
    for (int i = 0; i < p->count; i++) {
        p->nodeIds[p->count - i - 1] = current;
        current = closed[current]->parent;
    }
    return p;
}

void path_destroy(path *p) {
    free(p->nodeIds);
    free(p);
}

void closed_list_destroy(node_t **closed, int size) {
    #pragma omp parallel for
    for(int i = 0; i < size; i++) {
        if (closed[i] != NULL) free(closed[i]);
    }
    free(closed);
}

/**********************************************************************************************************/
/*                                            SPA* Algorithm                                              */
/**********************************************************************************************************/

int idle_threads = 0;
int total_threads = 0;
omp_lock_t idle_lock;

int terminate_condition() {
    int result = 0;
    omp_set_lock(&idle_lock);
    if (idle_threads >= total_threads) {
        result = 1;
    }
    omp_unset_lock(&idle_lock);
    return result;
}

path *astar_search(AStarSource *source, int start_id, int goal_id, int k, double *cpu_time_used) {

    omp_init_lock(&idle_lock);
    total_threads = k;

    heap_t *open = heap_init();
    omp_lock_t open_lock;
    omp_init_lock(&open_lock);

    node_t **closed = malloc(source->max_size * sizeof(node_t*));
    omp_lock_t *closed_locks = malloc(source->max_size * sizeof(omp_lock_t));

    for (int i = 0; i < source->max_size; i++) {
        closed[i] = NULL;
        omp_init_lock(&closed_locks[i]);
    }
    
    node_t *m = NULL;
    omp_lock_t m_lock;
    omp_init_lock(&m_lock);

    double start = omp_get_wtime();

    closed[start_id] = node_create(start_id, 0, source->heuristic(start_id, goal_id), -1);
    heap_insert(open, closed[start_id]);

    #pragma omp parallel num_threads(total_threads) shared(open, closed)
    {
        int steps = 0;
        neighbors_list *neighbors = neighbors_list_create();

        int tid = omp_get_thread_num();
        while(!terminate_condition()) {

            omp_set_lock(&open_lock);
            if (heap_is_empty(open) || (m != NULL && heap_min(open) >= m->fCost)) {
                omp_unset_lock(&open_lock);

                omp_set_lock(&idle_lock);
                idle_threads++;
                omp_unset_lock(&idle_lock);

                while (!terminate_condition()) {
                    #pragma omp flush
                }
                break;
            }
            
            node_t *current = heap_extract(open);
            omp_unset_lock(&open_lock);

            if (current == NULL) continue;

            omp_set_lock(&idle_lock);
            if (idle_threads > 0) idle_threads--;
            omp_unset_lock(&idle_lock);

            printf("Hilo %d, step %d, nodo actual: %d, fCost = %f\n", tid, steps++, current->id, current->fCost);
            
            if (current->id == goal_id) {
                int cost = current->gCost;
                omp_set_lock(&m_lock);
                if (m == NULL || cost < m->fCost) {
                    m = current;
                }
                omp_unset_lock(&m_lock);
                continue;
            }

            neighbors->count = 0;
            source->get_neighbors(neighbors, current->id);

            for(int i = 0; i < neighbors->count; i++) {
                int n_id = neighbors->nodeIds[i];
                float new_cost = closed[current->id]->gCost + neighbors->costs[i];
                if (closed[n_id]) {
                    if (new_cost < closed[n_id]->gCost) {
                        closed[n_id]->gCost = new_cost;
                        closed[n_id]->fCost = new_cost + source->heuristic(n_id, goal_id);
                        closed[n_id]->parent = current->id;
                        // printf("Actualiza nodo: %d con nuevo gCost = %f\n", n_id, new_cost);
                        if (closed[n_id]->is_open) {
                            omp_set_lock(&open_lock);
                            heap_update(open, closed[n_id]);
                            omp_unset_lock(&open_lock);
                        } else {
                            omp_set_lock(&open_lock);
                            heap_insert(open, closed[n_id]);
                            omp_unset_lock(&open_lock);
                        }
                    }
                    // printf("No actualiza nodo %d\n", n_id);
                } else {
                    // printf("Nuevo nodo %d con gCost = %f y fCost = %f\n", n_id, new_cost, new_cost + source->heuristic(n_id, goal_id));
                    closed[n_id] = node_create(n_id, new_cost, new_cost + source->heuristic(n_id, goal_id), current->id);
                    omp_set_lock(&open_lock);
                    heap_insert(open, closed[n_id]);
                    omp_unset_lock(&open_lock);
                    
                }
            }
        }

        neighbors_list_destroy(neighbors);
    }

    *cpu_time_used = omp_get_wtime() - start;
    
    path *p = retrace_path(closed, goal_id);
    
    closed_list_destroy(closed, source->max_size);
    for (int i = 0; i < source->max_size; i++) {
        omp_destroy_lock(&closed_locks[i]);
    }
    free(closed_locks);

    printf("Ok!");

    heap_destroy(open);
    omp_destroy_lock(&open_lock);
    omp_destroy_lock(&m_lock);

    printf("Ok!");

    return p;
}
