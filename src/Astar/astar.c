#include <stdio.h>
#include <omp.h>
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

path *retrace_path(node_t ** visited, int target) {
    path *p = malloc(sizeof(path));
    p->count = 0;
    p->cost = visited[target]->gCost;

    int current = target;
    while (visited[current]->parent != -1) {
        p->count++;
        current = visited[current]->parent;
    }

    p->nodeIds = malloc(p->count * sizeof(int));
    current = target;
    for (int i = 0; i < p->count; i++) {
        p->nodeIds[p->count - i - 1] = current;
        current = visited[current]->parent;
    }
    return p;
}

void path_destroy(path *p) {
    free(p->nodeIds);
    free(p);
}

void visited_list_destroy(node_t **visited, int size) {
    for(int i = 0; i < size; i++) {
        if (visited[i] != NULL) free(visited[i]);
    }
}

/**********************************************************************************************************/
/*                                              A* Algorithm                                              */
/**********************************************************************************************************/

path *astar_search(AStarSource *source, int start_id, int goal_id, double *cpu_time_used) {
    heap_t *open = heap_init();
    node_t **visited = malloc(source->max_size * sizeof(node_t*));
    for (int i = 0; i < source->max_size; i++) {
        visited[i] = NULL;
    }
    neighbors_list *neighbors = neighbors_list_create();
    
    visited[start_id] = node_create(start_id, 0, source->heuristic(start_id, goal_id), -1);
    heap_insert(open, visited[start_id]);

    node_t *current = NULL;
    
    double start = omp_get_wtime();

    int steps = 0;

    while(!heap_is_empty(open)) {
        // printf("Step: %d\n", ++steps);

        current = heap_extract(open);

        steps++;

        // printf("%d: Nodo actual: %d, fCost = %f\n", steps, current->id, current->fCost);

        if (current->id == goal_id) break;

        neighbors->count = 0;
        source->get_neighbors(neighbors, current->id);
        
        for(int i = 0; i < neighbors->count; i++) {
            int n_id = neighbors->nodeIds[i];
            float new_cost = visited[current->id]->gCost + neighbors->costs[i];
            if (visited[n_id]) { // Already visited
                if (new_cost < visited[n_id]->gCost) { // Better cost found
                    visited[n_id]->gCost = new_cost;
                    visited[n_id]->fCost = new_cost + source->heuristic(n_id, goal_id);
                    visited[n_id]->parent = current->id;
                    if (visited[n_id]->is_open) heap_update(open, visited[n_id]);   // Is in open
                    else heap_insert(open, visited[n_id]);  // Is not in open
                }
            } else { // New node
                visited[n_id] = node_create(n_id, new_cost, new_cost + source->heuristic(n_id, goal_id), current->id);
                heap_insert(open, visited[n_id]);
            }
        }
    }
    
    *cpu_time_used = omp_get_wtime() - start;

    // printf("Total de explansiones: %d\n", steps);

    path *p = NULL;
    if (current->id != goal_id) {
        printf("No se encontro el camino\n");
    } else {
        p = retrace_path(visited, goal_id);
    }
    visited_list_destroy(visited, source->max_size);
    heap_destroy(open);
    neighbors_list_destroy(neighbors);
    return p;
}