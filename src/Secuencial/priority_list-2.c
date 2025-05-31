#include <stdlib.h>
#include <float.h>
#include "astar.h"
#include "priority_list.h"

/* Crea una lista de prioridad */
priority_list *priority_list_create() {
    priority_list *list = malloc(sizeof(priority_list));
    list->count = 0;
    list->capacity = DEFAULT_QUEUE_SIZE;
    list->nodes = calloc(DEFAULT_QUEUE_SIZE + 1, sizeof(node*));
    return list;
}

/* Libera la memoria de la lista de prioridad */
void priority_list_destroy(priority_list *list) {
    free(list->nodes);
    free(list);
}

/* Intercambia dos nodos en la lista de prioridad y actualiza sus índices */
static void swap(node **n1, node **n2) {
    node *tmp = *n1;
    *n1 = *n2;
    *n2 = tmp;
    
    int index_tmp = (*(*n1)).open_index;
    (*n1)->open_index = (*n2)->open_index;
    (*n2)->open_index = index_tmp;
}


void priority_list_insert_or_update(priority_list *list, node *n) {
    if (list->count == list->capacity) {
        int new_capacity = list->capacity * 2;
        list->nodes = realloc(list->nodes, new_capacity * sizeof(node*));
        list->capacity = new_capacity;
    }

    if (!n->isOpen) {
        // INSERTAR NUEVO NODO
        list->count++;
        list->nodes[list->count] = n;
        n->open_index = list->count;
        
        int current = list->count;
        while (current > 1 && list->nodes[current]->fCost < list->nodes[current / 2]->fCost) {
            swap(&(list->nodes[current]), &(list->nodes[current / 2]));
            current /= 2;
        }
    } else {
        // ACTUALIZAR NODO EXISTENTE
        int current = n->open_index;
        while (current > 1 && list->nodes[current]->fCost < list->nodes[current / 2]->fCost) {
            swap(&(list->nodes[current]), &(list->nodes[current / 2]));
            current /= 2;
        }
    }
}

node *priority_list_extract(priority_list *list) {
    // Extraer el mínimo (siempre en la raíz)
    node *res = list->nodes[1];
    res->isOpen = 0;
    res->open_index = -1;

    // Reemplazar raíz con último elemento
    list->nodes[1] = list->nodes[list->count--];
    if (list->count > 0) {
        list->nodes[1]->open_index = 1;
    }

    // Heapify-down
    int current = 1;
    while(current < list->count) {
        int smallest = current;
        int child = 2 * current;
        if (child <= list->count && list->nodes[child]->fCost <= list->nodes[smallest]->fCost) {
            smallest = child;
        }
        child = 2 * current + 1;
        if (child <= list->count && list->nodes[child]->fCost <= list->nodes[smallest]->fCost) {
            smallest = child;
        }
        if (smallest == current) break;
        swap(&(list->nodes[current]), &(list->nodes[smallest]));
        current = smallest;
    }

    return res;
}

int priority_list_is_empty(priority_list *list) {
    return list ? list->count == 0 : 1;
}

double priority_list_get_min(priority_list *list) {
    if (!list || list->count == 0) return DBL_MAX;
    return list->nodes[1]->fCost;
}