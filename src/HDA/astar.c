#include "astar.h"
#include "heap.h"
#include <stdio.h>
#include <string.h>

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

path *retrace_path(node_t **closed, int target, int k) {
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
    q->nodes = malloc(q->capacity * sizeof(node_t*));
    omp_init_lock(&q->lock);
    return q;
}

void queue_destroy(queue_t *q) {
    omp_destroy_lock(&q->lock);
    free(q->nodes);
    free(q);
}

void enqueue(queue_t *q, node_t *n) {
    omp_set_lock(&q->lock);
    if (q->size == q->capacity) {
        q->capacity *= 2;
        q->nodes = realloc(q->nodes, q->capacity * sizeof(node_t*));
    }
    q->nodes[q->size++] = n;
    omp_unset_lock(&q->lock);
}

node_t *dequeue(queue_t *q) {
    node_t *res = NULL;
    omp_set_lock(&q->lock);
    if (q->size > 0) {
        res = q->nodes[--q->size];
    }
    omp_unset_lock(&q->lock);
    return res;
}

int *found;

int termination(int k) {
    for (int i = 0; i < k; i++) {
        if (found[i] == 0) return 0;
    }
    return 1;
}


/**********************************************************************************************************/
/*                                              A* Algorithm                                              */
/**********************************************************************************************************/

path *astar_search(AStarSource *source, int start_id, int goal_id, int k, double *cpu_time_used) {
    node_t **closed = malloc(source->max_size * sizeof(node_t*));
    heap_t **open = heaps_init(k);

    queue_t **queues = malloc(k * sizeof(queue_t*));
    neighbors_list **neighbors_lists = malloc(k * sizeof(neighbors_list*));
    for (int i = 0; i < k; i++) {
        queues[i] = queue_init();
        neighbors_lists[i] = neighbors_list_create();
    }
    
    closed[start_id] = node_create(start_id, 0, source->heuristic(start_id, goal_id), -1);
    source->get_neighbors(neighbors_lists[0], start_id);
    for (int i = 0; i < neighbors_lists[0]->count; i++) {
        int n_id = neighbors_lists[0]->nodeIds[i];
        float new_cost = neighbors_lists[0]->costs[i] + source->heuristic(start_id, n_id);
        enqueue(queues[hash(n_id, k)], node_create(n_id, new_cost, new_cost + source->heuristic(start_id, goal_id), start_id));
    }

    found = malloc(k * sizeof(int));
    for (int i = 0; i < k; i++) {
        found[i] = 0;
    }

    node_t *m = NULL;
    omp_lock_t m_lock;
    omp_init_lock(&m_lock);

    omp_set_num_threads(k);

    double start = omp_get_wtime();

    #pragma omp parallel num_threads(k)
    {
        int tid = omp_get_thread_num();

        while(!termination(k)) {
            node_t *msg;
            while((msg = dequeue(queues[tid])) != NULL) {
                // printf("Hilo %d: Recibe el nodo %d con gCost %.2f y fCost %.2f\n", tid, msg->id, msg->gCost, msg->fCost);
                if (closed[msg->id]) {
                    if (msg->gCost < closed[msg->id]->gCost) {
                        // printf("Hilo %d: Actualiza nodo %d con gCost %.2f y fCost %.2f\n", tid, msg->id, msg->gCost, msg->fCost);
                        closed[msg->id]->gCost = msg->gCost;
                        closed[msg->id]->fCost = msg->fCost;
                        closed[msg->id]->parent = msg->parent;
                        if (closed[msg->id]->is_open) heap_update(open[tid], closed[msg->id]);
                        else heap_insert(open[tid], closed[msg->id]);
                    }
                    free(msg);
                } else {
                    // printf("Hilo %d: Nodo %d es nuevo\n", tid, msg->id);
                    closed[msg->id] = msg;
                    heap_insert(open[tid], msg);
                }
            }

            if ((m != NULL && heap_min(open[tid]) >= m->fCost)) {
                found[tid] = 1;
                continue;
            }

            node_t *current = heap_extract(open[tid]);
            if (current == NULL) continue;

            // printf("Nodo actual: %d, fCost = %f\n", current->id, current->fCost);

            if (current->id == goal_id) {
                omp_set_lock(&m_lock);
                if (m == NULL || current->fCost < m->fCost) {
                    m = current;
                }
                omp_unset_lock(&m_lock);
                continue;
            }

            neighbors_lists[tid]->count = 0;
            source->get_neighbors(neighbors_lists[tid], current->id);

            for(int i = 0; i < neighbors_lists[tid]->count; i++) {
                int n_id = neighbors_lists[tid]->nodeIds[i];
                float new_cost = current->gCost + neighbors_lists[tid]->costs[i];
                int owner = hash(n_id, k);
                if (owner == tid) {
                    if (closed[n_id]) {
                        if (new_cost < closed[n_id]->gCost) {
                            closed[n_id]->gCost = new_cost;
                            closed[n_id]->fCost = new_cost + source->heuristic(n_id, goal_id);
                            closed[n_id]->parent = current->id;
                            if (closed[n_id]->is_open) heap_update(open[tid], closed[n_id]);
                            else heap_insert(open[tid], closed[n_id]);
                        }
                    } else {
                        closed[n_id] = node_create(n_id, new_cost, new_cost + source->heuristic(n_id, goal_id), current->id);
                        heap_insert(open[tid], closed[n_id]);
                    }
                } else {
                    enqueue(queues[owner], node_create(n_id, new_cost, new_cost + source->heuristic(n_id, goal_id), current->id));
                }
            }

        }
    }

    *cpu_time_used = omp_get_wtime() - start;

    path *p = retrace_path(closed, goal_id, k);
    closed_list_destroy(closed, source->max_size);
    heaps_destroy(open, k);
    for(int i = 0; i < k; i++) {
        queue_destroy(queues[i]);
        neighbors_list_destroy(neighbors_lists[i]);
    }
    free(queues);
    free(neighbors_lists);
    return p;
}