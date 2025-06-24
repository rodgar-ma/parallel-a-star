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
        q->elems = realloc(q->elems, q->capacity * sizeof(node_t*));
    }
    q->elems[q->size].node_id = elem.node_id;
    q->elems[q->size].gCost = elem.gCost;
    q->elems[q->size].parent_id = elem.parent_id;
    q->size++;
}

queue_elem_t dequeue(queue_t *q) {
    if (q->size > 0) {
        return q->elems[--q->size];
    }
    return (queue_elem_t){.node_id = -1, .gCost = -1, .parent_id = -1};
}

/**********************************************************************************************************/
/*                                              A* Algorithm                                              */
/**********************************************************************************************************/

inline int hasterminated(int *terminate, int k) {
    for (int i = 0; i < k; ++i) {
        if (terminate[i] == 0) {
            return 0;
        }
    }
    return 1;
}

path *astar_search(AStarSource *source, int start_id, int goal_id, int k, double *cpu_time_used) {
    node_t **closed = malloc(source->max_size * sizeof(node_t*));
    queue_t **queues = malloc(k * sizeof(queue_t*));
    int *terminate = malloc(k * sizeof(int));
    node_t *m = NULL;
    // float incumbent = FLT_MAX;
    for (int i = 0; i < k; i++) {
        queues[i] = queue_init();
        terminate[i] = 0;
    }
    
    closed[start_id] = node_create(start_id, 0, source->heuristic(start_id, goal_id), -1);

    if (start_id == goal_id) {
        path *p = retrace_path(closed, start_id);
        return p;
    }

    neighbors_list *neighbors = neighbors_list_create();
    source->get_neighbors(neighbors, start_id);
    for (int i = 0; i < neighbors->count; i++) {
        int n_id = neighbors->nodeIds[i];
        float new_cost = neighbors->costs[i] + source->heuristic(start_id, n_id);
        enqueue(queues[hash(n_id, k)], (queue_elem_t){.node_id = n_id, .gCost = new_cost, .parent_id = start_id});
    }
    neighbors_list_destroy(neighbors);

    double start = omp_get_wtime();
    int income_threshold = 10;

    #pragma omp parallel if(k>1) num_threads(k)
    {
        int tid = omp_get_thread_num();
        heap_t *open = heap_init();
        queue_elem_t *outgobuffer = malloc(INIT_NEIGHBORS_LIST_CAPACITY * sizeof(queue_elem_t));
        int buffer_max_size = INIT_NEIGHBORS_LIST_CAPACITY;
        int buffer_size = 0;
        queue_elem_t *buffer = malloc(buffer_size * sizeof(queue_elem_t));

        while(1) {
            queue_elem_t msg;

            // omp_set_lock(&queues[tid]->lock);

            if (queues[tid]->size > 0) {
                terminate[tid] = 0;
                if (queues[tid]->size > income_threshold) {
                    omp_set_lock(&queues[tid]->lock);
                    buffer_size = queues[tid]->size;
                    if (buffer_max_size < buffer_size) {
                        buffer_max_size = buffer_size;
                        buffer = realloc(buffer, buffer_size * sizeof(queue_elem_t));
                    }
                    memcpy(buffer, queues[tid]->elems, buffer_size * sizeof(queue_elem_t));
                    omp_unset_lock(&queues[tid]->lock);
                } else if (omp_test_lock(&queues[tid]->lock)) {
                    buffer_size = queues[tid]->size;
                    if (buffer_max_size < buffer_size) {
                        buffer_max_size = buffer_size;
                        buffer = realloc(buffer, buffer_size * sizeof(queue_elem_t));
                    }
                    memcpy(buffer, queues[tid]->elems, buffer_size * sizeof(queue_elem_t));
                    omp_unset_lock(&queues[tid]->lock);
                }
            }
                
            for (int i = 0; i < buffer_size; i++) {
                queue_elem_t msg = buffer[i];
                if (closed[msg.node_id]) {
                    if (msg.gCost < closed[msg.node_id]->gCost) {
                        // printf("Hilo %d: Actualiza nodo %d con gCost %.2f y fCost %.2f\n", tid, msg->id, msg->gCost, msg->fCost);
                        closed[msg.node_id]->gCost = msg.gCost;
                        closed[msg.node_id]->fCost = source->heuristic(msg.node_id, goal_id);
                        closed[msg.node_id]->parent = msg.parent_id;
                        if (closed[msg.node_id]->is_open) heap_update(open, closed[msg.node_id]);
                        else heap_insert(open, closed[msg.node_id]);
                    }
                } else {
                    // printf("Hilo %d: Nodo %d es nuevo\n", tid, msg.node_id);
                    closed[msg.node_id] = node_create(msg.node_id, msg.gCost, source->heuristic(msg.node_id, goal_id), msg.parent_id);
                    closed[msg.node_id]->is_open = 1;
                    heap_insert(open, closed[msg.node_id]);
                }
            }

            if (heap_is_empty(open) || (m != NULL && heap_min(open) < m->fCost)) {
                terminate[tid] = 1;
                if (hasterminated(terminate, k)) {
                    break;
                }
                continue;
            }
            
            node_t *current = heap_extract(open);

            if (current->id == goal_id) {
                #pragma omp critical
                {
                    if (current->fCost < m->fCost) {
                        m = current;
                    }
                }
            }

            neighbors->count = 0;
            source->get_neighbors(neighbors, current->id);

            for (int i = 0; i < neighbors->count; i++) {
                int n_id = neighbors->nodeIds[i];
                float new_cost = neighbors->costs[i];
                int owner = hash(n_id, k);

                if (owner == tid) {
                    
                }
            }

        }

        heap_destroy(open);
        neighbors_list_destroy(neighbors);
        free(outgobuffer);
        free(buffer);
    }

    *cpu_time_used = omp_get_wtime() - start;

    path *p = retrace_path(closed, m->id);
    closed_list_destroy(closed, source->max_size);
    for(int i = 0; i < k; i++) {
        queue_destroy(queues[i]);
    }
    free(queues);
    return p;
}