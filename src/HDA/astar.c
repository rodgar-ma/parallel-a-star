#include "astar.h"
#include "heap.h"

node_t *node_create(int id, float gCost, float fCost, int parent) {
    node_t *n = malloc(sizeof(node_t));
    n->id = id;
    n->parent = parent;
    n->gCost = gCost;
    n->fCost = fCost;
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
    for(int i = 0; i < size; i++) {
        if (closed[i] != NULL) free(closed[i]);
    }
}

int hash(int node, int k) {
    return node % k;
}

queue *queue_init() {
    queue *q = malloc(sizeof(queue));
    q->size = 0;
    q->capacity = INIT_QUEUE_CAPACITY;
    q->nodes = malloc(q->capacity * sizeof(heap_item_t*));
    omp_init_lock(&q->lock);
    return q;
}

void queue_destroy(queue *q) {
    free(q->nodes);
    omp_destroy_lock(&q->lock);
    free(q);
}

void enqueue(queue *q, heap_item_t *n) {
    omp_set_lock(&q->lock);
    if (q->size == q->capacity) {
        q->capacity *= 2;
        q->nodes = realloc(q->nodes, q->capacity * sizeof(node_t*));
    }
    q->nodes[q->size++] = n;
    omp_unset_lock(&q->lock);
}

heap_item_t *dequeue(queue *q) {
    heap_item_t *res = NULL;
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
    node_t **closed = malloc(source->max_size * sizeof(node_t*));

    heap_t *open[k];
    queue *q[k];
    neighbors_list *neighbors[k];
    for (int i = 0; i < k; i++) {
        open[i] = heap_init();
        q[i] = queue_init();
        neighbors[i] = neighbors_list_create(INIT_NEIGHBORS_LIST_CAPACITY);
    }
    
    closed[start_id] = node_create(start_id, 0, source->heuristic(start_id, goal_id), -1);
    enqueue(q[hash(start_id, k)], closed[start_id]);

    #pragma omp parallel num_threads(k)
    {
        int tid = omp_get_thread_num();
        int found = 0;

        while(!found) {
            heap_item_t * msg;
            while((msg = dequeue(q[tid])) != NULL) {
                if (!closed[msg->node] || closed[msg->node]->fCost < msg->fCost)
                heap_insert(open[tid], msg);
            }

            node_t *current = heap_extract(open[tid]);
            if (!current) continue;

            if (current->id = goal_id) {
                #pragma omp critical
                {
                    found = 1;
                }
                continue;
            }

            neighbors[tid]->count = 0;
            source->get_neighbors(neighbors[tid], current->id);

            for(int i = 0; i < neighbors[tid]->count; i++) {
                int id = neighbors[tid]->nodeIds[i];
                float cost = neighbors[tid]->costs[i];
                enqueue(q[hash(id, k)], );
        }
    }

    path *p = retrace_path(closed, goal_id);
    closed_list_destroy(closed, source->max_size);
    heap_destroy(open);
    neighbors_list_destroy(neighbors);
    return p;
}