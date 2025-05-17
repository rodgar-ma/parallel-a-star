#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include "astar.h"
#include "priority_list.h"

// Crea un nodo. Representa un estado.
node *node_create(astar_id_t id, double gCost, double fCost, node *parent) {
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
    list->nodeIds = calloc(list->capacity, sizeof(astar_id_t));
    return list;
}

// Libera la lista de vecinos.
void neighbors_list_destroy(neighbors_list *list) {
    free(list->nodeIds);
    free(list->costs);
    free(list);
}

// Función para el ususario. Añade un vecino a un nodo y su coste. Debe llamarse desde `get_neighbors`.
void add_neighbor(neighbors_list *neighbors, astar_id_t n_id, double cost) {
    if (neighbors->count == neighbors->capacity) {
        neighbors->capacity *= 2;
        neighbors->nodeIds = realloc(neighbors->nodeIds, neighbors->capacity * sizeof(astar_id_t));
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

    p->nodeIds = calloc(p->count, sizeof(astar_id_t));
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

path *find_path_sequential(AStarSource *source, astar_id_t s_id, astar_id_t t_id) {
    priority_list *open = priority_list_create();
    node** visited = calloc(source->max_size, sizeof(node*));
    neighbors_list *neighbors = neighbors_list_create();

    node *current = node_create(s_id, 0, source->heuristic(s_id, t_id), NULL);
    visited[s_id] = current;
    priority_list_insert_or_update(open, current);

    int num_expansiones = 0;

    while(!priority_list_is_empty(open)) {
        current = priority_list_extract(open);

        if (current->id == t_id) {
            break;
        }

        neighbors->count = 0;
        source->get_neighbors(neighbors, current->id);

        for (size_t i = 0; i < neighbors->count; i++) {
            astar_id_t neighbor_id = neighbors->nodeIds[i];
            double newCost = current->gCost + neighbors->costs[i];

            node *neighbor = visited[neighbor_id];
            num_expansiones++;

            if (!neighbor) {
                neighbor = node_create(neighbor_id, newCost, newCost + source->heuristic(neighbor_id, t_id), current);
                visited[neighbor_id] = neighbor;
            } else if (neighbor->isOpen && newCost < neighbor->gCost) {
                neighbor->gCost = newCost;
                neighbor->fCost = newCost + source->heuristic(neighbor_id, t_id);
                neighbor->parent = current;
            } else if (newCost + source->heuristic(neighbor_id, t_id) < neighbor->fCost) {
                neighbor->gCost = newCost;
                neighbor->fCost = newCost + source->heuristic(neighbor_id, t_id);
                neighbor->parent = current;
            } else {
                continue;
            }
            printf("Iteración: %d", steps);
            printf("Número de nodos visitados: %d\n", num_expansiones);
            priority_list_insert_or_update(open, neighbor);
        }
    }
    
    path *path = reatrace_path(current);
    printf("Número de expansiones: %d\n", num_expansiones);
    priority_list_destroy(open);
    neighbors_list_destroy(neighbors);
    free_visited(visited, source->max_size);
    return path;
}
