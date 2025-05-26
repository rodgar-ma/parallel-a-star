#include <stdlib.h>
#include <float.h>
#include <assert.h>
#include "astar.h"
#include "priority_list.h"

/* Crea una lista de prioridad */
priority_list *priority_list_create() {
    priority_list *list = malloc(sizeof(priority_list));
    if (!list) return NULL;
    
    list->capacity = DEFAULT_QUEUE_SIZE;
    list->count = 0;
    list->nodes = calloc(DEFAULT_QUEUE_SIZE, sizeof(node*));
    
    if (!list->nodes) {
        free(list);
        return NULL;
    }
    
    return list;
}

/* Libera la memoria de la lista de prioridad */
void priority_list_destroy(priority_list *list) {
    if (!list) return;
    free(list->nodes);
    free(list);
}

/* Intercambia dos nodos en la lista de prioridad y actualiza sus índices */
static void swap(priority_list *list, int i, int j) {
    node *temp = list->nodes[i];
    list->nodes[i] = list->nodes[j];
    list->nodes[j] = temp;
    
    // Actualizar índices
    list->nodes[i]->open_index = i;
    list->nodes[j]->open_index = j;
}

// static void swap(node **n1,  node **n2) {
// 	node *tmp = *n1;
// 	*n1 = *n2;
// 	*n2 = tmp;
// }

void priority_list_insert_or_update(priority_list *list, node *n) {
    if (!list || !n) return;

    if (!n->isOpen) {
        // INSERTAR NUEVO NODO
        if (list->count == list->capacity) {
            int new_capacity = list->capacity * 2;
            node **new_nodes = realloc(list->nodes, new_capacity * sizeof(node*));
            if (!new_nodes) return;
            
            list->nodes = new_nodes;
            list->capacity = new_capacity;
        }

        int i = list->count++;
        list->nodes[i] = n;
        n->open_index = i;
        n->isOpen = 1;

        // Heapify-up
        while (i > 0) {
            int parent = (i - 1) / 2;
            if (list->nodes[parent]->fCost <= list->nodes[i]->fCost) break;
            
            swap(list, i, parent);
            i = parent;
        }
    } else {
        // ACTUALIZAR NODO EXISTENTE
        int i = n->open_index;

        // Heapify-up (reposiciona el nodo si su fCost ha mejorado)
        while (i > 0) {
            int parent = (i - 1) / 2;
            if (list->nodes[parent]->fCost <= n->fCost) break;
            
            swap(list, i, parent);
            i = parent;
        }
    }
}

node *priority_list_extract(priority_list *list) {
    if (!list || list->count == 0) return NULL;

    // Extraer el mínimo (siempre en la raíz)
    node *minNode = list->nodes[0];
    minNode->isOpen = 0;
    minNode->open_index = -1;
    
    // Reemplazar raíz con último elemento
    list->nodes[0] = list->nodes[--list->count];
    if (list->count > 0) {
        list->nodes[0]->open_index = 0;
    }

    // Heapify-down
    int i = 0;
    while (1) {
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        int smallest = i;

        if (left < list->count && list->nodes[left]->fCost <= list->nodes[smallest]->fCost) {
            smallest = left;
        }
        if (right < list->count && list->nodes[right]->fCost <= list->nodes[smallest]->fCost) {
            smallest = right;
        }
        if (smallest == i) break;

        swap(list, i, smallest);
        i = smallest;
    }

    return minNode;
}

int priority_list_is_empty(priority_list *list) {
    return list ? list->count == 0 : 1;
}

double priority_list_get_min(priority_list *list) {
    if (!list || list->count == 0) return DBL_MAX;
    return list->nodes[0]->fCost;
}