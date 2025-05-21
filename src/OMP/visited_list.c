#include <stdlib.h>
#include "visited_list.h"

visited_list *visited_list_create(int capacity) {
    visited_list *H = malloc(sizeof(visited_list));
    H->capacity = capacity;
    H->nodes = calloc(capacity, sizeof(node*));
    H->locks = malloc(capacity * sizeof(omp_lock_t));
    for (int i = 0; i < capacity; i++) {
        omp_init_lock(&H->locks[i]);
    }
    return H;
}

void visited_list_destroy(visited_list *H) {
    for (int i = 0; i < H->capacity; i++) {
        omp_destroy_lock(&H->locks[i]);
    }
    free(H->locks);
    free(H->nodes);
    free(H);
}

int visited_list_contains(visited_list *H, int node_id) {
    return H->nodes[node_id] != NULL;
}

void visited_list_insert(visited_list *H, list *S, int index, double hCost) {
    int id = S->ids[index];
    omp_set_lock(&H->locks[id]);
    if (H->nodes[id] == NULL) {
        H->nodes[id] = node_create(id, S->gCosts[index], S->gCosts[index] + hCost, S->parents[index]);
    } else if (H->nodes[id]->gCost <= S->gCosts[index]) {
        H->nodes[id]->gCost = S->gCosts[index];
        H->nodes[id]->fCost = S->gCosts[index] + hCost;
        H->nodes[id]->parent = S->parents[index];
    }
    omp_unset_lock(&H->locks[id]);
}

int visited_list_is_better(visited_list *H, list *S, int index) {
    return H->nodes[S->ids[index]]->gCost <= S->gCosts[index];
}