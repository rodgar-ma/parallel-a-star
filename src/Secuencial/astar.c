#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include <time.h>
#include "astar.h"
#include "priority_list.h"

// Crea un nodo. Representa un estado.
node *node_create(int id, double gCost, double fCost, node *parent) {
    node *n = malloc(sizeof(node));
    n->id = id;
    n->parent = parent;
    n->gCost = gCost;
    n->fCost = fCost;
    n->isOpen = 0;
    n->open_index = -1;
    return n;
}

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
path *reatrace_path(node *target) {
    path *p = malloc(sizeof(path));
    p->count = 0;
    p->cost = target->fCost;

    node *current = target;
    while (current) {
        p->count++;
        current = current->parent;
    }

    p->nodeIds = calloc(p->count, sizeof(int));
    current = target;
    for (int i = 0; i < p->count; i++) {
        p->nodeIds[p->count - i - 1] = current->id;
        current = current->parent;
    }
    return p;
}

// Libera la memoria del camino.
void path_destroy(path *p) {
    free(p->nodeIds);
    free(p);
}

// Libera los nodos visitados.
void free_visited(node **visited, size_t size) {
    for(int i = 0; i < size; i++) {
        if (visited[i]) free(visited[i]);
    }
    free(visited);
}

/**********************************************************************************************************/
/*                                              A* Algorithm                                              */
/**********************************************************************************************************/

path *find_path_sequential(AStarSource *source, int s_id, int t_id, double *cpu_time_used) {
    priority_list *open = priority_list_create();
    node** visited = calloc(source->max_size, sizeof(node*));
    neighbors_list *neighbors = neighbors_list_create();

    node *current = node_create(s_id, 0, source->heuristic(s_id, t_id), NULL);
    visited[s_id] = current;
    priority_list_insert_or_update(open, current);

    int n_iters = 0;

    clock_t start = clock();
    while(!priority_list_is_empty(open)) {
        current = priority_list_extract(open);

        if (current->id == t_id) {
            break;
        }

        neighbors->count = 0;
        source->get_neighbors(neighbors, current->id);
        for (size_t i = 0; i < neighbors->count; i++) {
            int neighbor_id = neighbors->nodeIds[i];
            double newfCost = current->gCost + neighbors->costs[i] + source->heuristic(neighbor_id, t_id);

            if (!visited[neighbor_id]) {
                visited[neighbor_id] = node_create(neighbor_id, current->gCost + neighbors->costs[i], newfCost, current);;
                priority_list_insert_or_update(open, visited[neighbor_id]);
            } else if (newfCost < visited[neighbor_id]->fCost) {
                visited[neighbor_id]->gCost = current->gCost + neighbors->costs[i];
                visited[neighbor_id]->fCost = newfCost;
                visited[neighbor_id]->parent = current;
                priority_list_insert_or_update(open, visited[neighbor_id]);
            }
        }
        n_iters++;
    }
    clock_t end = clock();
    printf("%d iteraciones.\n", n_iters);
    *cpu_time_used = (double)(end - start) / CLOCKS_PER_SEC;
    path *path = reatrace_path(current);
    priority_list_destroy(open);
    neighbors_list_destroy(neighbors);
    free_visited(visited, source->max_size);
    return path;
}
