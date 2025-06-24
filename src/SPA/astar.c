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

path *astar_search(AStarSource *source, int start_id, int goal_id, int k, double *cpu_time_used) {

    heap_t *open = heap_init();
    omp_lock_t open_lock;
    omp_init_lock(&open_lock);

    node_t **closed = malloc(source->max_size * sizeof(node_t*));
    omp_lock_t *closed_locks = malloc(source->max_size * sizeof(omp_lock_t));

    #pragma omp parallel for
    for (int i = 0; i < source->max_size; i++) {
        closed[i] = NULL;
        omp_init_lock(&closed_locks[i]);
    }

    printf("Comienza búsqueda...\n");
    
    node_t *m = NULL;
    omp_lock_t m_lock;
    omp_init_lock(&m_lock);

    closed[start_id] = node_create(start_id, 0, source->heuristic(start_id, goal_id), -1);
    heap_insert(open, closed[start_id]);

    double start;

    #pragma omp parallel if(k>1) num_threads(k) shared(open, closed, cpu_time_used)
    {
        int steps = 0;
        neighbors_list *neighbors = neighbors_list_create();
        int tid = omp_get_thread_num();

        #pragma omp master
        {
            start = omp_get_wtime();
        }

        while(1) {
            
            float m_cost;
            omp_set_lock(&m_lock);
            m_cost = (m != NULL) ? m->fCost : FLT_MAX;
            omp_unset_lock(&m_lock);

            // printf("Hilo %d, m_cost = %f\n", tid, m_cost);

            omp_set_lock(&open_lock);
            if (heap_min(open) >= m_cost) {
                omp_unset_lock(&open_lock);
                break;
            }
            if (heap_is_empty(open)) {
                omp_unset_lock(&open_lock);
                omp_set_lock(&m_lock);
                int found = (m != NULL);
                omp_unset_lock(&m_lock);
                if (found) break;
                else continue;
            }

            node_t *current = heap_extract(open);
            omp_unset_lock(&open_lock);

            if (current == NULL) continue;

            steps++;

            // printf("Hilo %d, step %d, nodo actual: %d, fCost = %f\n", tid, steps++, current->id, current->fCost);
            
            if (current->id == goal_id) {
                omp_set_lock(&m_lock);
                if (m == NULL || current->fCost < m->fCost) {
                    m = current;
                }
                omp_unset_lock(&m_lock);
                continue;
            }

            neighbors->count = 0;
            source->get_neighbors(neighbors, current->id);

            for(int i = 0; i < neighbors->count; i++) {
                int n_id = neighbors->nodeIds[i];
                omp_set_lock(&closed_locks[n_id]);
                float new_cost = closed[current->id]->gCost + neighbors->costs[i];
                omp_unset_lock(&closed_locks[n_id]);
                if (closed[n_id]) {
                    if (new_cost < closed[n_id]->gCost) {
                        omp_set_lock(&closed_locks[n_id]);
                        if (new_cost < closed[n_id]->gCost) {
                            closed[n_id]->gCost = new_cost;
                            closed[n_id]->fCost = new_cost + source->heuristic(n_id, goal_id);
                            closed[n_id]->parent = current->id;
                            omp_set_lock(&open_lock);
                            if (closed[n_id]->is_open) {
                                heap_update(open, closed[n_id]);
                            } else {
                                heap_insert(open, closed[n_id]);
                            }
                            omp_unset_lock(&open_lock);
                        }
                        omp_unset_lock(&closed_locks[n_id]);
                        // printf("Actualiza nodo: %d con nuevo gCost = %f\n", n_id, new_cost);
                    }
                    // printf("No actualiza nodo %d\n", n_id);
                } else {
                    // printf("Nuevo nodo %d con gCost = %f y fCost = %f\n", n_id, new_cost, new_cost + source->heuristic(n_id, goal_id));
                    omp_set_lock(&closed_locks[n_id]);
                    closed[n_id] = node_create(n_id, new_cost, new_cost + source->heuristic(n_id, goal_id), current->id);
                    omp_set_lock(&open_lock);
                    heap_insert(open, closed[n_id]);
                    omp_unset_lock(&open_lock);
                    omp_unset_lock(&closed_locks[n_id]);
                    
                }
            }
        }

        #pragma omp master
        {
            *cpu_time_used = omp_get_wtime() - start;
        }

        printf("Hilo %d. Total de expansiones: %d\n", tid, steps);

        neighbors_list_destroy(neighbors);
    }

    // printf("Camino encontrado!\n");
    
    path *p = retrace_path(closed, goal_id);
    
    closed_list_destroy(closed, source->max_size);

    for (int i = 0; i < source->max_size; i++) {
        omp_destroy_lock(&closed_locks[i]);
    }
    free(closed_locks);

    heap_destroy(open);
    omp_destroy_lock(&m_lock);
    omp_destroy_lock(&open_lock);

    return p;
}
