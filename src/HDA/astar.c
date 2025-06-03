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

int hash(int node, int k) {
    return node % k;
}

path *retrace_path(node_t ** closed[], int target, int k) {
    path *p = malloc(sizeof(path));
    p->count = 0;
    p->cost = closed[hash(target, k)][target]->gCost;

    int current = target;
    while (closed[hash(current, k)][current]->parent != -1) {
        p->count++;
        current = closed[hash(current, k)][current]->parent;
    }

    p->nodeIds = malloc(p->count * sizeof(int));
    current = target;
    for (int i = 0; i < p->count; i++) {
        p->nodeIds[p->count - i - 1] = current;
        current = closed[hash(current, k)][current]->parent;
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
    memset(q->nodes, 0, q->capacity * sizeof(node_t*));
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
        int old_size = q->size;
        q->capacity *= 2;
        q->nodes = realloc(q->nodes, q->capacity * sizeof(node_t*));
        memset(q->nodes + old_size, 0, (q->capacity - old_size) * sizeof(node_t*));
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

/**********************************************************************************************************/
/*                                              A* Algorithm                                              */
/**********************************************************************************************************/

path *astar_search(AStarSource *source, int start_id, int goal_id, int k) {
    node_t ***closed = malloc(k * sizeof(node_t**));
    heap_t **open = malloc(k * sizeof(heap_t*));
    queue_t **q = malloc(k * sizeof(queue_t*));
    neighbors_list **neighbors = malloc(k * sizeof(neighbors_list*));
    omp_lock_t *closed_locks = malloc(k * sizeof(omp_lock_t));
    for (int i = 0; i < k; i++) {
        closed[i] = malloc(source->max_size * sizeof(node_t*));
        open[i] = heap_init();
        q[i] = queue_init();
        neighbors[i] = neighbors_list_create();
        omp_init_lock(&closed_locks[i]);
    }
    
    enqueue(q[hash(start_id, k)], node_create(start_id, 0, source->heuristic(start_id, goal_id), -1));

    int found = 0;

    omp_set_num_threads(k);

    #pragma omp parallel shared(source, start_id, goal_id, k, closed, open, q, neighbors, found, closed_locks)
    {
        int tid = omp_get_thread_num();
        
        #pragma omp flush(found)
        while(!found) {
            node_t *msg;
            while((msg = dequeue(q[tid])) != NULL) {
                printf("Hilo %d: Actualizando nodo %d con gCost %.2f y fCost %.2f\n", tid, msg->id, msg->gCost, msg->fCost);
                omp_set_lock(&closed_locks[tid]);
                if (closed[tid][msg->id]) {
                    if (msg->gCost < closed[tid][msg->id]->gCost) {
                        closed[tid][msg->id]->id = msg->id;
                        closed[tid][msg->id]->gCost = msg->gCost;
                        closed[tid][msg->id]->fCost = msg->fCost;
                        closed[tid][msg->id]->parent = msg->parent;
                        heap_insert(open[tid], closed[tid][msg->id]);
                    }
                    free(msg);
                } else {
                    closed[tid][msg->id] = msg;
                    heap_insert(open[tid], msg);
                }
                omp_unset_lock(&closed_locks[tid]);
            }

            node_t *current = heap_extract(open[tid]);
            if (!current) continue;

            if (current->id == 262143) {
                printf("\n");
            }

            if (current->id == goal_id) {
                #pragma omp critical
                {
                    found = 1;
                    #pragma omp flush(found)
                }
                continue;
            }

            neighbors[tid]->count = 0;
            source->get_neighbors(neighbors[tid], current->id);

            for(int i = 0; i < neighbors[tid]->count; i++) {
                int id = neighbors[tid]->nodeIds[i];
                float new_cost = current->gCost + neighbors[tid]->costs[i];
                enqueue(q[hash(id, k)], node_create(id, new_cost, new_cost + source->heuristic(id, goal_id), current->id));
            }
        }
    }

    path *p = retrace_path(closed, goal_id, k);
    for(int i = 0; i < k; i++) {
        closed_list_destroy(closed[i], source->max_size);
        heap_destroy(open[i]);
        queue_destroy(q[i]);
        neighbors_list_destroy(neighbors[i]);
        omp_destroy_lock(&closed_locks[i]);
    }
    free(closed);
    free(open);
    free(q);
    free(neighbors);
    free(closed_locks);
    return p;
}