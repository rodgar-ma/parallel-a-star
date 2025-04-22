#include <stdlib.h>
#include <float.h>
#include "astar.h"
#include "open_list.h"

/* Crea un conjunto de listas de prioridad */
open_list **open_lists_create(int k) {
    open_list **lists = calloc(k, sizeof(open_list*));
    for (int i = 0; i < k; i++) {
        lists[i] = open_list_create();
    }
    return lists;
}

/* Crea una lista de prioridad */
open_list *open_list_create() {
    open_list *list = malloc(sizeof(open_list));
    list->capacity = DEFAULT_QUEUE_SIZE;
    list->count = 0;
    list->nodes = calloc(DEFAULT_QUEUE_SIZE, sizeof(node*));
    return list;
}

void open_lists_destroy(open_list **lists, int k) {
    for (int i = 0; i < k; i++) {
        open_list_destroy(lists[i]);
    }
    free(lists);
}

/* Libera la memoria de la lista de prioridad */
void open_list_destroy(open_list *list) {
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
void open_list_insert_or_update(open_list *list, node *n) {
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
node *open_list_extract(open_list *list) {
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

/* Devuelve 1 si la lista está vacía, 0 en caso contrario */
int open_list_is_empty(open_list *list) {
    return list->count == 0;
}

/* Devuelve 1 si todas las listas están vacías, 0 en caso contrario */
int open_lists_are_empty(open_list **lists, int k) {
    for (int i = 0; i < k; i++) {
        if (!open_list_is_empty(lists[i])) return 0;
    }
    return 1;
}

/* Devuelve el mínimo fCost de una lista */
double open_list_get_min(open_list *list) {
    if (open_list_is_empty(list)) return DBL_MAX;
    return list->nodes[0]->fCost;
}

/* Devuelve el mínimo fCost de un conjunto de listas */
double open_lists_get_min(open_list **list, int k) {
    double min_f = DBL_MAX;
    for (int i = 0; i < k; i++) {
        if (open_list_is_empty(list[i])) continue;
        double new_f = open_list_get_min(list[i]);
        if (new_f < min_f) min_f = new_f;
    }
    return min_f;
}