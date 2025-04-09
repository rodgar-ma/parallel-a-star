#include <stdlib.h>
#include <float.h>
#include "astar.h"
#include "priority_queue.h"

#define DEFAULT_QUEUE_SIZE 16

priority_queue **priority_queues_create(int k) {
    priority_queue **pqs = calloc(k, sizeof(priority_queue*));
    for (int i = 0; i < k; i++) {
        pqs[i] = priority_queue_create();
    }
    return pqs;
}

priority_queue *priority_queue_create() {
    priority_queue *pq = malloc(sizeof(priority_queue));
    pq->capacity = DEFAULT_QUEUE_SIZE;
    pq->count = 0;
    pq->nodes = calloc(DEFAULT_QUEUE_SIZE, sizeof(node*));
    return pq;
}

void priority_queue_destroy(priority_queue *pq) {
    free(pq->nodes);
    free(pq);
}

static void swap(node **a, node **b) {
    node *temp = *a;
    *a = *b;
    *b = temp;
}

void priority_queue_insert(priority_queue *pq, node *n) {
    if (pq->count == pq->capacity) {
        pq->capacity = 1 + (2 * pq->capacity);
        pq->nodes = realloc(pq->nodes, pq->capacity * sizeof(node*));
    }

    size_t i = pq->count++;
    pq->nodes[i] = n;

    while (i > 0 && pq->nodes[(i-1)/2]->fCost > pq->nodes[i]->fCost) {
        swap(&pq->nodes[i], &pq->nodes[(i-1)/2]);
        i = (i - 1) / 2;
    }
}

node *priority_queue_extract(priority_queue *pq) {
    if (pq->count == 0) return NULL;
    node *minNode = pq->nodes[0];
    pq->nodes[0] = pq->nodes[--pq->count];

    size_t i = 0;
    while (2 * i + 1 < pq->count) {
        size_t left = 2 * i + 1;
        size_t right = 2 * i + 2;
        size_t smallest = left;

        if (right < pq->count && pq->nodes[right]->fCost < pq->nodes[left]->fCost) {
            smallest = right;
        }

        if (pq->nodes[i]->fCost <= pq->nodes[smallest]->fCost) break;

        swap(&pq->nodes[i], &pq->nodes[smallest]);
        i = smallest;
    }
    return minNode;
}

int priority_queue_is_empty(priority_queue *pq) {
    return pq->count == 0;
}

int priority_queues_are_empty(priority_queue **pqs, int k) {
    for (int i = 0; i < k; i++) {
        if (!priority_queue_is_empty(pqs[i])) return 0;
    }
    return 1;
}

double priority_queue_get_min(priority_queue *pq) {
    if (priority_queue_is_empty(pq)) return DBL_MAX;
    return pq->nodes[0]->fCost;
}

double priority_queues_get_min(priority_queue **pq, int k) {
    double min_f = DBL_MAX;
    for (int i = 0; i < k; i++) {
        if (priority_queue_is_empty(pq[i])) continue;
        double new_f = priority_queue_get_min(pq[i]);
        if (new_f < min_f) min_f = new_f;
    }
    return min_f;
}