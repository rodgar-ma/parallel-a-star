#include <stdlib.h>
#include <omp.h>
#include "visited_list.h"

visited_list *visited_list_create(int capacity) {
    visited_list *H = malloc(sizeof(visited_list));
    H->capacity = capacity;
    H->nodes = calloc(capacity, sizeof(node*));
    H->locks = calloc(capacity, sizeof(omp_lock_t));
    for (int i = 0; i < capacity; i++) {
       omp_init_lock(&H->locks[i]);
    }
    return H;
}

void visited_list_destroy(visited_list *H) {

    for (int i = 0; i < H->capacity; i++) {
        if (H->nodes[i] != NULL) free(H->nodes[i]);
        //omp_destroy_lock(&H->locks[i]);
    }
    free(H->locks);
    free(H->nodes);
    free(H);
}

int visited_list_contains(visited_list *H, int node_id) {
    return H->nodes[node_id] != NULL;
}

int visited_list_insert(visited_list *H, int id, double gCost, double fCost, node *parent) {
    // omp_set_lock(&H->locks[id]);
    if (H->nodes[id] == NULL) {
        H->nodes[id] = node_create(id, gCost, fCost, parent);
        // omp_unset_lock(&H->locks[id]);
        return 1;
    } else if (gCost < H->nodes[id]->gCost) {
        H->nodes[id]->gCost = gCost;
        H->nodes[id]->fCost = fCost;
        H->nodes[id]->parent = parent;
        // omp_unset_lock(&H->locks[id]);
        return 1;
    }
    // omp_unset_lock(&H->locks[id]);
    return 0;
}
