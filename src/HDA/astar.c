#include "astar.h"
#include "heap.h"
#include <stdio.h>
#include <string.h>
#include <float.h>

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

neighbors_list *neighbors_list_create(void) {
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

static inline int hash(int node, int k) {
    return node % k;
}

path *retrace_path(node_t **closed, int target) {
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
    for(int i = 0; i < size; i++) {
        if (closed[i] != NULL) free(closed[i]);
    }
    free(closed);
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

path *astar_search(AStarSource *source, int start_id, int goal_id, int k, double *cpu_time_used) {
    node_t **closed = calloc(source->max_size, sizeof(node_t*));
    queue_t **queues = malloc(k * sizeof(queue_t*));
    int *terminate = malloc(k * sizeof(int));
    node_t *m = NULL;
    // float incumbent = FLT_MAX;
    for (int i = 0; i < k; i++) {
        queues[i] = queue_init();
        terminate[i] = 0;
    }
    
    double start = omp_get_wtime();

    closed[start_id] = node_create(start_id, 0, source->heuristic(start_id, goal_id), -1);

    if (start_id == goal_id) {
        *cpu_time_used = omp_get_wtime() - start;
        path *p = retrace_path(closed, start_id);
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

    int income_threshold = 10;

    #pragma omp parallel if(k>1) num_threads(k) shared(closed, queues, terminate, m, start)
    {
        int tid = omp_get_thread_num();
        heap_t *open = heap_init();
        neighbors_list * neighbors = neighbors_list_create();
        buffer_t *incomebuffer = buffer_init();
        buffer_t **outgobuffer = malloc(k * sizeof(buffer_t*));
        for (int i = 0; i < k; i++) {
            outgobuffer[i] = buffer_init();
        }

        int steps = 0;

        #pragma omp master
        {
            start = omp_get_wtime();
        }

        while(1) {

            steps++;

            if (queues[tid]->size > 0) {
                terminate[tid] = 0;
                if (queues[tid]->size > income_threshold) {
                    omp_set_lock(&queues[tid]->lock);
                    fill_buffer(incomebuffer, queues[tid]);
                    omp_unset_lock(&queues[tid]->lock);
                } else if (omp_test_lock(&queues[tid]->lock)) {
                    fill_buffer(incomebuffer, queues[tid]);
                    omp_unset_lock(&queues[tid]->lock);
                }
            }
            
            if (incomebuffer->size > 0) {
                for (int i = 0; i < incomebuffer->size; i++) {
                    queue_elem_t msg = incomebuffer->elems[i];
                    if (closed[msg.node_id] != NULL) {
                        if (msg.gCost < closed[msg.node_id]->gCost) {
                            closed[msg.node_id]->gCost = msg.gCost;
                            closed[msg.node_id]->fCost = msg.gCost + source->heuristic(msg.node_id, goal_id);
                            closed[msg.node_id]->parent = msg.parent_id;
                            if (closed[msg.node_id]->is_open) heap_update(open, closed[msg.node_id]);
                            else heap_insert(open, closed[msg.node_id]);
                        }
                    } else {
                        closed[msg.node_id] = node_create(msg.node_id, msg.gCost, msg.gCost + source->heuristic(msg.node_id, goal_id), msg.parent_id);
                        heap_insert(open, closed[msg.node_id]);
                    }
                }
                incomebuffer->size = 0;
            }

            if (heap_is_empty(open) || (m != NULL && heap_min(open) >= m->fCost)) {
                terminate[tid] = 1;
                if (hasterminated(terminate, k) && m != NULL) {
                    break;
                }
                for (int i = 0; i < k; i++) {
                    if (i != tid && outgobuffer[i]->size > 0) {
                        if (omp_test_lock(&queues[i]->lock)) {
                            fill_queue(outgobuffer[i], queues[i]);
                            omp_unset_lock(&queues[i]->lock);
                            outgobuffer[i]->size = 0;
                        }
                    }
                }
                continue;
            }
            
            node_t *current = heap_extract(open);

            if (current->id == goal_id) {
                #pragma omp critical
                {
                    if (m == NULL || current->fCost < m->fCost) {
                        m = current;
                    }
                }
                continue;
            }

            neighbors->count = 0;
            source->get_neighbors(neighbors, current->id);

            for (int i = 0; i < neighbors->count; i++) {
                int n_id = neighbors->nodeIds[i];
                float new_cost = current->gCost + neighbors->costs[i];
                int owner = hash(n_id, k);

                if (owner == tid) {
                    if (closed[n_id]) {
                        if (new_cost < closed[n_id]->gCost) {
                            closed[n_id]->gCost = new_cost;
                            closed[n_id]->fCost = new_cost + source->heuristic(n_id, goal_id);
                            closed[n_id]->parent = current->id;
                            if (closed[n_id]->is_open) heap_update(open, closed[n_id]);
                            else heap_insert(open, closed[n_id]);
                        }
                    } else {
                        closed[n_id] = node_create(n_id, new_cost, new_cost + source->heuristic(n_id, goal_id), current->id);
                        heap_insert(open, closed[n_id]);
                    }
                } else if (omp_test_lock(&queues[owner]->lock)) {
                    enqueue(queues[owner], (queue_elem_t){n_id, new_cost, current->id});
                    omp_unset_lock(&queues[owner]->lock);
                } else {
                    buffer_insert(outgobuffer[owner], (queue_elem_t){n_id, new_cost, current->id});
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
            buffer_destroy(outgobuffer[i]);
        }
        free(outgobuffer);
    }

    

    path *p = retrace_path(closed, m->id);
    closed_list_destroy(closed, source->max_size);
    for(int i = 0; i < k; i++) {
        queue_destroy(queues[i]);
    }
    free(queues);
    free (terminate);
    return p;
}