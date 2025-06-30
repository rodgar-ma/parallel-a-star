#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <omp.h>
#include "hdastar.h"
#include "heap.h"
#include "buffer.h"

static inline int hash(int node, int k) {
    return node % k;
}


/**********************************************************************************************************/
/*                                             HDA* Algorithm                                             */
/**********************************************************************************************************/

static inline int hasterminated(int *terminate, int k) {
    for (int i = 0; i < k; ++i) {
        if (terminate[i] == 0) {
            return 0;
        }
    }
    return 1;
}

path *hdastar_search(AStarSource *source, int start_id, int goal_id, int k, double *cpu_time_used) {
    
    node_t **visited = calloc(source->max_size, sizeof(node_t*));
    buffer_t **incomebuffers = malloc(k * sizeof(buffer_t*));
    omp_lock_t *incomebuffers_locks = malloc(k * sizeof(omp_lock_t));
    int *terminate = malloc(k * sizeof(int));
    omp_lock_t terminate_lock;
    omp_init_lock(&terminate_lock);
    node_t *m = NULL;
    omp_lock_t m_lock;
    omp_init_lock(&m_lock);

    int *nodes = malloc(k * sizeof(int));

    #pragma omp parallel for
    for (int i = 0; i < k; i++) {
        incomebuffers[i] = buffer_init();
        omp_init_lock(&incomebuffers_locks[i]);
        terminate[i] = 0;
        nodes[i] = 0;
    }
    
    double start = omp_get_wtime();

    visited[start_id] = node_create(start_id, 0, source->heuristic(start_id, goal_id), -1);
    nodes[hash(start_id, k)]++;

    if (start_id == goal_id) {
        *cpu_time_used = omp_get_wtime() - start;
        path *p = retrace_path(visited, start_id);
        return p;
    }

    neighbors_list *neighbors = neighbors_list_create();
    source->get_neighbors(neighbors, start_id);
    for (int i = 0; i < neighbors->count; i++) {
        int n_id = neighbors->nodeIds[i];
        float new_cost = neighbors->costs[i];
        buffer_insert(incomebuffers[hash(n_id, k)], (buffer_elem_t){n_id, new_cost, start_id});
    }
    neighbors_list_destroy(neighbors);

    int income_threshold = 1000;
    int outgo_threshold = 1000;

    #pragma omp parallel num_threads(k) shared(visited, incomebuffers, incomebuffers_locks, terminate, m, income_threshold)
    {
        int tid = omp_get_thread_num();
        heap_t *open = heap_init();
        neighbors_list * neighbors = neighbors_list_create();
        buffer_t *tmp_buffer = buffer_init();
        buffer_t **outgobuffers = malloc(k * sizeof(buffer_t*));
        for (int i = 0; i < k; i++) {
            outgobuffers[i] = buffer_init();
        }

        int steps = 0;

        #pragma omp master
        {
            start = omp_get_wtime();
        }

        while(1) {
            
            // Check income buffer
            if (incomebuffers[tid]->size > 0) {
                omp_set_lock(&terminate_lock);
                terminate[tid] = 0;
                omp_unset_lock(&terminate_lock);
                if (incomebuffers[tid]->size >= income_threshold) {
                    omp_set_lock(&incomebuffers_locks[tid]);
                    fill_buffer(tmp_buffer, incomebuffers[tid]);
                    omp_unset_lock(&incomebuffers_locks[tid]);
                } else if (omp_test_lock(&incomebuffers_locks[tid])) {
                    fill_buffer(tmp_buffer, incomebuffers[tid]);
                    omp_unset_lock(&incomebuffers_locks[tid]);
                }
            }
            
            // Fill open list
            if (tmp_buffer->size > 0) {
                for (int i = 0; i < tmp_buffer->size; i++) {
                    buffer_elem_t msg = tmp_buffer->elems[i];
                    if (visited[msg.node_id] != NULL) {
                        if (msg.gCost < visited[msg.node_id]->gCost) {
                            visited[msg.node_id]->gCost = msg.gCost;
                            visited[msg.node_id]->fCost = msg.gCost + source->heuristic(msg.node_id, goal_id);
                            visited[msg.node_id]->parent = msg.parent_id;
                            if (visited[msg.node_id]->is_open) heap_update(open, visited[msg.node_id]);
                            else heap_insert(open, visited[msg.node_id]);
                        }
                    } else {
                        visited[msg.node_id] = node_create(msg.node_id, msg.gCost, msg.gCost + source->heuristic(msg.node_id, goal_id), msg.parent_id);
                        nodes[tid]++;
                        heap_insert(open, visited[msg.node_id]);
                    }
                }
                tmp_buffer->size = 0;
            }

            // Get best actual path cost
            omp_set_lock(&m_lock);
            float m_cost = (m != NULL) ? m->fCost : FLT_MAX;
            omp_unset_lock(&m_lock);

            // Check if open is empty or next node has higher f than actual path
            if (heap_is_empty(open) || heap_min(open) >= m_cost) {
                omp_set_lock(&terminate_lock);
                terminate[tid] = 1;
                if (hasterminated(terminate, k) && m != NULL) {
                    omp_unset_lock(&terminate_lock);
                    break;
                }
                omp_unset_lock(&terminate_lock);

                for (int i = 0; i < k; i++) {
                    if (i != tid && outgobuffers[i]->size > 0) {
                        if (omp_test_lock(&incomebuffers_locks[i])) {
                            fill_buffer(incomebuffers[i], outgobuffers[i]);
                            omp_unset_lock(&incomebuffers_locks[i]);
                            outgobuffers[i]->size = 0;
                        }
                    }
                }
                continue;
            }
            
            // Get next node from open
            node_t *current = heap_extract(open);
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

            for (int i = 0; i < neighbors->count; i++) {
                int n_id = neighbors->nodeIds[i];
                float new_cost = current->gCost + neighbors->costs[i];
                int owner = hash(n_id, k);
                if (owner == tid) {
                    if (visited[n_id]) {
                        if (new_cost < visited[n_id]->gCost) {
                            visited[n_id]->gCost = new_cost;
                            visited[n_id]->fCost = new_cost + source->heuristic(n_id, goal_id);
                            visited[n_id]->parent = current->id;
                            if (visited[n_id]->is_open) heap_update(open, visited[n_id]);
                            else heap_insert(open, visited[n_id]);
                        }
                    } else {
                        visited[n_id] = node_create(n_id, new_cost, new_cost + source->heuristic(n_id, goal_id), current->id);
                        nodes[tid]++;
                        heap_insert(open, visited[n_id]);
                    }
                } else if (outgobuffers[owner]->size > outgo_threshold) {
                    omp_set_lock(&incomebuffers_locks[owner]);
                    buffer_insert(incomebuffers[owner], (buffer_elem_t){n_id, new_cost, current->id});
                    fill_buffer(incomebuffers[owner], outgobuffers[owner]);
                    omp_unset_lock(&incomebuffers_locks[owner]);
                } else if (omp_test_lock(&incomebuffers_locks[owner])) {
                    buffer_insert(incomebuffers[owner], (buffer_elem_t){n_id, new_cost, current->id});
                    if (outgobuffers[owner]->size > 0) {
                        fill_buffer(incomebuffers[owner], outgobuffers[owner]);
                    }
                    omp_unset_lock(&incomebuffers_locks[owner]);
                } else {
                    buffer_insert(outgobuffers[owner], (buffer_elem_t){n_id, new_cost, current->id});
                }
            }

        }

        #pragma omp master
        {
            *cpu_time_used = omp_get_wtime() - start;
        }

        heap_destroy(open);
        neighbors_list_destroy(neighbors);
        buffer_destroy(tmp_buffer);
        for (int i = 0; i < k; i++) {
            buffer_destroy(outgobuffers[i]);
        }
        free(outgobuffers);
    }

    int total_nodes = 0;
    for (int i = 0; i < k; i++) {
        total_nodes += nodes[i];
    }

    free(nodes);
    // printf("Total de nodos generados: %d\n", total_nodes);

    path *p = retrace_path(visited, m->id);
    visited_list_destroy(visited, source->max_size);
    #pragma omp parallel for
    for(int i = 0; i < k; i++) {
        buffer_destroy(incomebuffers[i]);
        omp_destroy_lock(&incomebuffers_locks[i]);
    }
    free(incomebuffers);
    free(incomebuffers_locks);
    free (terminate);
    omp_destroy_lock(&terminate_lock);
    omp_destroy_lock(&m_lock);
    return p;
}