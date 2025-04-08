#ifndef PRIORITY_QUEUE_H
#define PRIORITY_QUEUE_H

#include <stdlib.h>
#include "astar.h"

typedef struct __priority_queue priority_queue;

struct __priority_queue {
    size_t capacity;
    size_t count;
    node **nodes;
};

priority_queue **priority_queues_create(int k);
priority_queue *priority_queue_create();
void priority_queue_destroy(priority_queue *pq);
void priority_queue_insert(priority_queue *pq, node *node);
node *priority_queue_extract(priority_queue *pq);
int priority_queue_is_empty(priority_queue *pq);
int priority_queues_are_empty(priority_queue **pq);
double priority_queue_get_min(priority_queue *pq);
double priority_queues_get_min(priority_queue **pq, int k);

#endif