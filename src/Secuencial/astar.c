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
    for(int i = 0; i < size; i++) {
        if (closed[i] != NULL) free(closed[i]);
    }
}

/**********************************************************************************************************/
/*                                              A* Algorithm                                              */
/**********************************************************************************************************/

path *astar_search(AStarSource *source, int start_id, int goal_id) {
    heap_t *open = heap_init();
    node_t **closed = malloc(source->max_size * sizeof(node_t*));
    neighbors_list *neighbors = neighbors_list_create();
    
    closed[start_id] = node_create(start_id, 0, source->heuristic(start_id, goal_id), -1);
    heap_insert(open, closed[start_id]);

    while(!heap_is_empty(open)) {
        node_t *current = heap_extract(open);

        if (current->id == goal_id) break;

        neighbors->count = 0;
        source->get_neighbors(neighbors, current->id);
        for(int i = 0; i < neighbors->count; i++) {
            int n_id = neighbors->nodeIds[i];
            float new_cost = closed[current->id]->gCost + neighbors->costs[i];
            if (closed[n_id]) {
                if (new_cost < closed[n_id]->gCost) {
                    closed[n_id]->gCost = new_cost;
                    closed[n_id]->fCost = new_cost + source->heuristic(n_id, goal_id);
                    closed[n_id]->parent = current->id;
                    if (closed[n_id]->is_open) heap_update(open, closed[n_id]);
                    else heap_insert(open, closed[n_id]);
                }
            } else {
                closed[n_id] = node_create(n_id,new_cost, new_cost + source->heuristic(n_id, goal_id), current->id);
                heap_insert(open, closed[n_id]);
            }
        }
    }

    path *p = retrace_path(closed, goal_id);
    closed_list_destroy(closed, source->max_size);
    heap_destroy(open);
    neighbors_list_destroy(neighbors);
    return p;
}