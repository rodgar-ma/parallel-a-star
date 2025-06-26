#include <stdio.h>
#include <omp.h>
#include <float.h>
#include "spastar.h"
#include "heap.h"

/**********************************************************************************************************/
/*                                            SPA* Algorithm                                              */
/**********************************************************************************************************/

path *spastar_search(AStarSource *source, int start_id, int goal_id, int k, double *cpu_time_used) {

    heap_t *open = heap_init();
    omp_lock_t open_lock;
    omp_init_lock(&open_lock);

    node_t **visited = malloc(source->max_size * sizeof(node_t*));
    omp_lock_t *visited_locks = malloc(source->max_size * sizeof(omp_lock_t));

    #pragma omp parallel for
    for (int i = 0; i < source->max_size; i++) {
        visited[i] = NULL;
        omp_init_lock(&visited_locks[i]);
    }
    
    // Best current goal
    node_t *m = NULL;
    omp_lock_t m_lock;
    omp_init_lock(&m_lock);

    // Termination counter
    int terminated = 0;
    omp_lock_t terminated_lock;
    omp_init_lock(&terminated_lock);

    visited[start_id] = node_create(start_id, 0, source->heuristic(start_id, goal_id), -1);
    heap_insert(open, visited[start_id]);

    double start;

    #pragma omp parallel if(k > 1) num_threads(k) shared(open, open_lock, visited, visited_locks, m, m_lock, terminated, terminated_lock)
    {
        int steps = 0;
        neighbors_list *neighbors = neighbors_list_create();
        int tid = omp_get_thread_num();
        int waiting = 0;

        #pragma omp master
        {
            start = omp_get_wtime();
        }

        while(1) {
            
            // Check if all threads have terminated
            omp_set_lock(&terminated_lock);
            if (terminated == k) {
                omp_unset_lock(&terminated_lock);
                break;
            }
            omp_unset_lock(&terminated_lock);

            // Get the best actual path cost
            omp_set_lock(&m_lock);
            float m_cost = (m != NULL) ? m->fCost : FLT_MAX;
            omp_unset_lock(&m_lock);

            // Check if open is empty or next node has higher f than actual path
            omp_set_lock(&open_lock);
            if (heap_is_empty(open) || heap_min(open) >= m_cost) {
                omp_unset_lock(&open_lock);
                if (!waiting) {
                    waiting = 1;
                    omp_set_lock(&terminated_lock);
                    terminated++;
                    omp_unset_lock(&terminated_lock);
                }
                continue;
            }
            
            if (waiting) {
                waiting = 0;
                omp_set_lock(&terminated_lock);
                terminated--;
                omp_unset_lock(&terminated_lock);
            }

            // Get next node from open
            node_t *current = heap_extract(open);
            omp_unset_lock(&open_lock);
            steps++;
            
            // Goal found
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
                float new_cost = current->gCost + neighbors->costs[i];
                if (visited[n_id]) {
                    if (new_cost < visited[n_id]->gCost) {
                        omp_set_lock(&visited_locks[n_id]);
                        if (new_cost < visited[n_id]->gCost) {
                            visited[n_id]->gCost = new_cost;
                            visited[n_id]->fCost = new_cost + source->heuristic(n_id, goal_id);
                            visited[n_id]->parent = current->id;
                            omp_set_lock(&open_lock);
                            if (visited[n_id]->is_open) {
                                heap_update(open, visited[n_id]);
                            } else {
                                heap_insert(open, visited[n_id]);
                            }
                            omp_unset_lock(&open_lock);
                        }
                        omp_unset_lock(&visited_locks[n_id]);
                        // printf("Actualiza nodo: %d con nuevo gCost = %f\n", n_id, new_cost);
                    }
                    // printf("No actualiza nodo %d\n", n_id);
                } else {
                    // printf("Nuevo nodo %d con gCost = %f y fCost = %f\n", n_id, new_cost, new_cost + source->heuristic(n_id, goal_id));
                    omp_set_lock(&visited_locks[n_id]);
                    visited[n_id] = node_create(n_id, new_cost, new_cost + source->heuristic(n_id, goal_id), current->id);
                    omp_set_lock(&open_lock);
                    heap_insert(open, visited[n_id]);
                    omp_unset_lock(&open_lock);
                    omp_unset_lock(&visited_locks[n_id]);
                    
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
    
    path *p = retrace_path(visited, goal_id);
    
    visited_list_destroy(visited, source->max_size);

    for (int i = 0; i < source->max_size; i++) {
        omp_destroy_lock(&visited_locks[i]);
    }
    free(visited_locks);

    heap_destroy(open);
    omp_destroy_lock(&m_lock);
    omp_destroy_lock(&open_lock);

    return p;
}
