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
    int h = node % k;
    return (h >= 0 && h < k) ? h : 0;
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

/**********************************************************************************************************/
/*                                              A* Algorithm                                              */
/**********************************************************************************************************/

path *astar_search(AStarSource *source, int start_id, int goal_id, int k, double *cpu_time_used) {
    node_t **closed = malloc(source->max_size * sizeof(node_t*));
    #pragma omp parallel for
    for (int i = 0; i < source->max_size; i++) {
        closed[i] = NULL;
    }

    queue_t **queues = malloc(k * sizeof(queue_t*));
    for (int i = 0; i < k; i++) {
        queues[i] = queue_init();
    }
    
    enqueue(queues[hash(start_id, k)], node_create(start_id, 0, source->heuristic(start_id, goal_id), -1));

    int found = 0;

    double start = omp_get_wtime();

    #pragma omp parallel num_threads(k)
    {
        int tid = omp_get_thread_num();
        neighbors_list *neighbors = neighbors_list_create();
        heap_t *open = heap_init();

        while(!found) {
            node_t *msg;
            while((msg = dequeue(queues[tid])) != NULL) {
                // printf("Hilo %d: Obtiene el nodo %d de la cola con gCost %.2f y fCost %.2f\n", tid, msg->id, msg->gCost, msg->fCost);
                if (closed[msg->id]) {
                    if (msg->gCost < closed[msg->id]->gCost) {
                        // printf("Hilo %d: Actualizando nodo %d con gCost %.2f y fCost %.2f\n", tid, msg->id, msg->gCost, msg->fCost);
                        closed[msg->id]->id = msg->id;
                        closed[msg->id]->gCost = msg->gCost;
                        closed[msg->id]->fCost = msg->fCost;
                        closed[msg->id]->parent = msg->parent;
                        if (closed[msg->id]->is_open) heap_update(open, closed[msg->id]);
                        else heap_insert(open, closed[msg->id]);
                    }
                    free(msg);
                } else {
                    // printf("Hilo %d: insertando nodo %d con gCost %.2f y fCost %.2f\n", tid, msg->id, msg->gCost, msg->fCost);
                    closed[msg->id] = msg;
                    heap_insert(open, msg);
                }
            }

            node_t *current = heap_extract(open);
            if (current == NULL) continue;
            
            // #pragma omp critical
            // {
            //     printf("Hilo %d: nodo actual %d, step = %d\n", tid, current->id, ++step);
            // }

            if (current->id == goal_id) {
                #pragma omp critical
                {
                    found = 1;
                }
                continue;
            }

            neighbors->count = 0;
            source->get_neighbors(neighbors, current->id);

            for(int i = 0; i < neighbors->count; i++) {
                int id = neighbors->nodeIds[i];
                float new_cost = current->gCost + neighbors->costs[i];
                // printf("Hilo %d envia nodo %d a hilo %d\n", tid, id, hash(id, k));
                enqueue(queues[hash(id, k)], node_create(id, new_cost, new_cost + source->heuristic(id, goal_id), current->id));
            }

        }

        neighbors_list_destroy(neighbors);
        heap_destroy(open);
    }

    *cpu_time_used = omp_get_wtime() - start;

    path *p = retrace_path(closed, goal_id, k);
    closed_list_destroy(closed, source->max_size);
    for(int i = 0; i < k; i++) {
        queue_destroy(queues[i]);
    }
    free(queues);
    return p;
}