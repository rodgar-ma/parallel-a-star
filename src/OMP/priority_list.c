#include <stdlib.h>
#include <float.h>
#include "astar.h"
#include "priority_list.h"

/* Crea un conjunto de listas de prioridad */
priority_list **priority_lists_create(int k) {
    priority_list **lists = calloc(k, sizeof(priority_list*));
    for(int i = 0; i < k; i++) {
        lists[i] = priority_list_create();
    }
    return lists;
}

/* Crea una lista de prioridad */
priority_list *priority_list_create() {
    priority_list *list = malloc(sizeof(priority_list));
    list->capacity = DEFAULT_QUEUE_SIZE;
    list->count = 0;
    list->nodes = calloc(DEFAULT_QUEUE_SIZE, sizeof(node*));
    return list;
}

/* Libera la memoria de un conjunto de listas de prioridad */
void priority_lists_destroy(priority_list **lists, int k) {
    for (int i = 0; i < k; i++) {
        priority_list_destroy(lists[i]);
    }
    free(lists);
}

/* Libera la memoria de la lista de prioridad */
void priority_list_destroy(priority_list *list) {
    free(list->nodes);
    free(list);
}

/* Intercambia dos nodos en la lista de prioridad */
static void swap(node **a, node **b) {
    node *temp = *a;
    *a = *b;
    *b = temp;
}

/* Inserta un nodo en la lista de prioridad o actualiza su valor si ya existe. Si la lista está llena, se duplica su tamaño */
void priority_list_insert_or_update(priority_list *list, node *n) {
    for (size_t i = 0; i < list->count; i++) {
        if (list->nodes[i]->id == n->id) {
            if (n->fCost < list->nodes[i]->fCost) {
                list->nodes[i] = n;
                while (i > 0 && list->nodes[(i-1)/2]->fCost > list->nodes[i]->fCost) {
                    swap(&list->nodes[i], &list->nodes[(i-1)/2]);
                    i = (i - 1) / 2;
                }
            }
            return;
        }
    }

    if (list->count == list->capacity) {
        list->capacity = 1 + (2 * list->capacity);
        list->nodes = realloc(list->nodes, list->capacity * sizeof(node*));
    }

    size_t i = list->count++;
    list->nodes[i] = n;

    while (i > 0 && list->nodes[(i-1)/2]->fCost > list->nodes[i]->fCost) {
        swap(&list->nodes[i], &list->nodes[(i-1)/2]);
        i = (i - 1) / 2;
    }
}

/* Extrae el nodo con menor fCost de la lista y lo devuelve. Si la lista está vacía, devuelve NULL */
node *priority_list_extract(priority_list *list) {
    if (list->count == 0) return NULL;
    node *minNode = list->nodes[0];
    list->nodes[0] = list->nodes[--list->count];

    size_t i = 0;
    while (2 * i + 1 < list->count) {
        size_t left = 2 * i + 1;
        size_t right = 2 * i + 2;
        size_t smallest = left;

        if (right < list->count && list->nodes[right]->fCost < list->nodes[left]->fCost) {
            smallest = right;
        }

        if (list->nodes[i]->fCost <= list->nodes[smallest]->fCost) break;

        swap(&list->nodes[i], &list->nodes[smallest]);
        i = smallest;
    }
    return minNode;
}

/* Devuelve 1 si todas las listas están vacías, 0 en caso contrario */
int priority_lists_are_empty(priority_list **lists, int k) {
    for (int i = 0; i < k; i++) {
        if (!priority_list_is_empty(lists[i])) return 0;
    }
    return 1;
}

/* Devuelve 1 si la lista está vacía, 0 en caso contrario */
int priority_list_is_empty(priority_list *list) {
    return list->count == 0;
}

/* Devuelve el mínimo fCost de una conjunto de listas */
double priority_lists_get_min(priority_list **lists, int k) {
    double min = DBL_MAX;
    for (int i = 0; i < k; i++) {
        double tmp = priority_list_get_min(lists[i]);
        if (tmp < min) min = tmp;
    }
    return min;
}

/* Devuelve el mínimo fCost de una lista */
double priority_list_get_min(priority_list *list) {
    if (priority_list_is_empty(list)) return DBL_MAX;
    return list->nodes[0]->fCost;
}
