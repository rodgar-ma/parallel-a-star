#include <stdio.h>
#include <limits.h>
#include <float.h>
#include <time.h>
#include <omp.h>
#include "astar.h"
#include "priority_list.h"
#include "MapUtils.h"

// Crea un nodo. Representa un estado.
// node *node_create(int id, double gCost, double fCost, node *parent) {
//     node *n = malloc(sizeof(node));
//     n->id = id;
//     n->parent = parent;
//     n->gCost = gCost;
//     n->fCost = fCost;
//     n->isOpen = 0;
//     n->open_index = -1;
//     return n;
// }

// Lista de vecinos de un nodo.
neighbors_list *neighbors_list_create() {
    neighbors_list *list = malloc(sizeof(neighbors_list));
    list->capacity = DEFAULT_NIEGHBORS_LIST_CAPACITY;
    list->count = 0;
    list->costs = calloc(list->capacity, sizeof(double));
    list->nodeIds = calloc(list->capacity, sizeof(int));
    return list;
}

// Libera la lista de vecinos.
void neighbors_list_destroy(neighbors_list *list) {
    free(list->nodeIds);
    free(list->costs);
    free(list);
}

// Función para el ususario. Añade un vecino a un nodo y su coste. Debe llamarse desde `get_neighbors`.
void add_neighbor(neighbors_list *neighbors, int n_id, double cost) {
    if (neighbors->count == neighbors->capacity) {
        neighbors->capacity *= 2;
        neighbors->nodeIds = realloc(neighbors->nodeIds, neighbors->capacity * sizeof(int));
        neighbors->costs = realloc(neighbors->costs, neighbors->capacity * sizeof(double));
    }
    neighbors->nodeIds[neighbors->count] = n_id;
    neighbors->costs[neighbors->count] = cost;
    neighbors->count++;
}

// Devuelve el camino desde el nodo inicial hasta el nodo `target`.
path *retrace_path(int target, double* cost_so_far, int *came_from) {
    path *p = malloc(sizeof(path));
    p->count = 0;
    p->cost = cost_so_far[target];

    int current = target;
    while (came_from[current] != current) {
        p->count++;
        current = came_from[current];
    }

    p->nodeIds = calloc(p->count, sizeof(int));
    current = target;
    for (int i = 0; i < p->count; i++) {
        p->nodeIds[p->count - i - 1] = current;
        current = came_from[current];
    }
    return p;
}

// Libera la memoria del camino.
void path_destroy(path *p) {
    free(p->nodeIds);
    free(p);
}

// Libera los nodos visitados.
void free_visited(node **visited, int size) {
    for(int i = 0; i < size; i++) {
        if (visited[i]) free(visited[i]);
    }
    free(visited);
}

/**********************************************************************************************************/
/*                                              A* Algorithm                                              */
/**********************************************************************************************************/

path *find_path_sequential(AStarSource *source, int s_id, int t_id, double *cpu_time_used) {
    // priority_list *open = priority_list_create();
    // node** visited = calloc(source->max_size, sizeof(node*));
    neighbors_list *neighbors = neighbors_list_create();

    // node *current = node_create(s_id, 0, source->heuristic(s_id, t_id), NULL);
    // visited[s_id] = current;
    // priority_list_insert_or_update(open, current);

    priority_list *frontier = priority_list_create(DEFAULT_PRIORITY_LIST_CAPACITY);
    int *came_from = calloc(source->max_size, sizeof(int));
    double *cost_so_far = calloc(source->max_size, sizeof(double));
    for(int i = 0; i < source->max_size; i++) {
        cost_so_far[i] = DBL_MAX;
        came_from[i] = -1;
    }

    int current = s_id;
    
    priority_list_insert(frontier, current, 0);
    came_from[current] = current;
    cost_so_far[current] = 0;

    int n_iters = 0;
    double start = omp_get_wtime();
    while(frontier->size > 0) {
        current = priority_list_extract(frontier);

        // printf("STEP %d\n", n_iters);
        // Node n = GetNodeById(MAP, current->id);
        // Node p;
        // if (current->parent != NULL){
        //     p = GetNodeById(MAP, current->parent->id);
        //     printf("(%d, %d)->(%d, %d)\n", p->x, p->y, n->x, n->y);
        // } else {
        //     printf("(%d, %d)\n", n->x, n->y);
        // }

        if (current == t_id) {
            break;
        }

        neighbors->count = 0;
        source->get_neighbors(neighbors, current);
        for (int i = 0; i < neighbors->count; i++) {
            double newCost = cost_so_far[current] + neighbors->costs[i]; //  + source->heuristic(neighbor_id, t_id);
            // Node neighbor = GetNodeById(MAP, neighbor_id);
            if (newCost < cost_so_far[neighbors->nodeIds[i]]) {
                cost_so_far[neighbors->nodeIds[i]] = newCost;
                priority_list_insert(frontier, neighbors->nodeIds[i], newCost + source->heuristic(neighbors->nodeIds[i], t_id));
                came_from[neighbors->nodeIds[i]] = current;
            }
        }
                // priority_list_insert_or_update(open, visited[neighbor_id]);
                // printf("Nuevo nodo: (%d, %d), gCost = %f, fCost = %f\n", neighbor->x, neighbor->y, visited[neighbor_id]->gCost, visited[neighbor_id]->fCost);
            // } else if (newCost < visited[neighbor_id]->gCost) {
                // printf("Nodo actualizado: (%d, %d), old_gCost = %f, new_gCost = %f\n", neighbor->x, neighbor->y, visited[neighbor_id]->gCost, newCost);
                // visited[neighbor_id]->gCost = newCost;
                // visited[neighbor_id]->fCost = newCost + source->heuristic(neighbor_id, t_id);
                // visited[neighbor_id]->parent = current;
                // priority_list_insert_or_update(open, visited[neighbor_id]);
            // } // else {
                // printf("Nodo (%d, %d) no se actualiza\n", neighbor->x, neighbor->y);
            // }
        // }
        // printf("\n\n");
        n_iters++;
    }
    double end = omp_get_wtime();
    *cpu_time_used = (end - start) * 1000.0;
    printf("%d iteraciones.\n", n_iters);
    path *path = retrace_path(current, cost_so_far, came_from);
    priority_list_destroy(frontier);
    neighbors_list_destroy(neighbors);
    free(came_from);
    free(cost_so_far);
    // free_visited(visited, source->max_size);
    return path;
}
