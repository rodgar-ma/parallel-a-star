#include <stdio.h>
#include <string.h>
#include <float.h>
#include "hdastar.h"
#include "heap.h"


static inline int hash(int node, int k) {
    return node % k;
}


// QUEUES

queue_t *queue_init(void) {
    queue_t *q = malloc(sizeof(queue_t));
    q->size = 0;
    q->capacity = INIT_QUEUE_CAPACITY;
    q->elems = malloc(q->capacity * sizeof(queue_elem_t));
    omp_init_lock(&q->lock);
    return q;
}

void queue_destroy(queue_t *q) {
    omp_destroy_lock(&q->lock);
    free(q->elems);
    free(q);
}

void enqueue(queue_t *q, queue_elem_t elem) {
    if (q->size == q->capacity) {
        q->capacity *= 2;
        q->elems = realloc(q->elems, q->capacity * sizeof(queue_elem_t));
    }
    q->elems[q->size++] = elem;
}

queue_elem_t dequeue(queue_t *q) {
    if (q->size > 0) {
        return q->elems[--q->size];
    }
    return (queue_elem_t){.node_id = -1, .gCost = -1, .parent_id = -1};
}

// BUFFERS

buffer_t *buffer_init(void) {
    buffer_t *buffer = malloc(sizeof(buffer_t));
    buffer->size = 0;
    buffer->capacity = INIT_QUEUE_CAPACITY;
    buffer->elems = malloc(buffer->capacity * sizeof(queue_elem_t));
    return buffer;
}

void buffer_destroy(buffer_t *buffer) {
    free(buffer->elems);
    free(buffer);
}

void buffer_insert(buffer_t *buffer, queue_elem_t elem) {
    if (buffer->size == buffer->capacity) {
        buffer->capacity *= 2;
        buffer->elems = realloc(buffer->elems, buffer->capacity * sizeof(queue_elem_t));
    }
    buffer->elems[buffer->size++] = elem;
}

void fill_buffer(buffer_t *buffer, queue_t *queue) {
    if (buffer->capacity < buffer->size + queue->size) {
        buffer->capacity = buffer->size + queue->size;
        buffer->elems = realloc(buffer->elems, buffer->capacity * sizeof(queue_elem_t));
    }
    memcpy(buffer->elems + buffer->size, queue->elems, queue->size * sizeof(queue_elem_t));
    buffer->size += queue->size;
    queue->size = 0;
}

void fill_queue(buffer_t *buffer, queue_t *queue) {
    if (queue->capacity < queue->size + buffer->size) {
        queue->capacity = queue->size + buffer->size;
        queue->elems = realloc(queue->elems, queue->capacity * sizeof(queue_elem_t));
    }
    memcpy(queue->elems + queue->size, buffer->elems, buffer->size * sizeof(queue_elem_t));
    queue->size +=  buffer->size;
    buffer->size = 0;
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
    queue_t **queues = malloc(k * sizeof(queue_t*));
    int *terminate = malloc(k * sizeof(int));
    omp_lock_t terminate_lock;
    omp_init_lock(&terminate_lock);
    node_t *m = NULL;
    omp_lock_t m_lock;
    omp_init_lock(&m_lock);

    for (int i = 0; i < k; i++) {
        queues[i] = queue_init();
        terminate[i] = 0;
    }
    
    double start = omp_get_wtime();

    visited[start_id] = node_create(start_id, 0, source->heuristic(start_id, goal_id), -1);

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
        enqueue(queues[hash(n_id, k)], (queue_elem_t){n_id, new_cost, start_id});
    }
    neighbors_list_destroy(neighbors);

    int income_threshold = 5;
    int outgo_threshold = 5;

    #pragma omp parallel if(k > 1) num_threads(k) shared(visited, queues, terminate, m, income_threshold)
    {
        int tid = omp_get_thread_num();
        heap_t *open = heap_init();
        neighbors_list * neighbors = neighbors_list_create();
        buffer_t *incomebuffer = buffer_init();
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
            if (queues[tid]->size > 0) {
                omp_set_lock(&terminate_lock);
                terminate[tid] = 0;
                omp_unset_lock(&terminate_lock);
                if (queues[tid]->size >= income_threshold) {
                    omp_set_lock(&queues[tid]->lock);
                    fill_buffer(incomebuffer, queues[tid]);
                    omp_unset_lock(&queues[tid]->lock);
                } else if (omp_test_lock(&queues[tid]->lock)) {
                    fill_buffer(incomebuffer, queues[tid]);
                    omp_unset_lock(&queues[tid]->lock);
                }
            }
            
            // Fill open list
            if (incomebuffer->size > 0) {
                for (int i = 0; i < incomebuffer->size; i++) {
                    queue_elem_t msg = incomebuffer->elems[i];
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
                        heap_insert(open, visited[msg.node_id]);
                    }
                }
                incomebuffer->size = 0;
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
                        if (omp_test_lock(&queues[i]->lock)) {
                            fill_queue(outgobuffers[i], queues[i]);
                            omp_unset_lock(&queues[i]->lock);
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
                        heap_insert(open, visited[n_id]);
                    }
                } else if (outgobuffers[owner]->size > outgo_threshold) {
                    omp_set_lock(&queues[owner]->lock);
                    enqueue(queues[owner], (queue_elem_t){n_id, new_cost, current->id});
                    fill_queue(outgobuffers[owner], queues[owner]);
                    omp_unset_lock(&queues[owner]->lock);
                } else if (omp_test_lock(&queues[owner]->lock)) {
                    enqueue(queues[owner], (queue_elem_t){n_id, new_cost, current->id});
                    if (outgobuffers[owner]->size > 0) {
                        fill_queue(outgobuffers[owner], queues[owner]);
                    }
                    omp_unset_lock(&queues[owner]->lock);
                } else {
                    buffer_insert(outgobuffers[owner], (queue_elem_t){n_id, new_cost, current->id});
                }
            }

        }

        #pragma omp master
        {
            *cpu_time_used = omp_get_wtime() - start;
        }

        heap_destroy(open);
        neighbors_list_destroy(neighbors);
        buffer_destroy(incomebuffer);
        for (int i = 0; i < k; i++) {
            buffer_destroy(outgobuffers[i]);
        }
        free(outgobuffers);
    }

    

    path *p = retrace_path(visited, m->id);
    visited_list_destroy(visited, source->max_size);
    for(int i = 0; i < k; i++) {
        queue_destroy(queues[i]);
    }
    free(queues);
    free (terminate);
    return p;
}