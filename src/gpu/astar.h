#ifndef ASTAR_H
#define ASTAR_H

#include <stdlib.h>

const int OPEN_LIST_SIZE = 10000000;
const int NODE_LIST_SIZE = 150000000;
const int ANSWER_LIST_SIZE = 10000000;

const int NUM_BLOCK = 13 * 3;
const int NUM_THREAD = 192;
const int NUM_TOTAL = NUM_BLOCK * NUM_THREAD;

const int VALUE_PER_THREAD = 1;
const int NUM_VALUE = NUM_TOTAL * VALUE_PER_THREAD;

const int HEAP_CAPACITY = OPEN_LIST_SIZE / NUM_TOTAL;

typedef struct {
    u_int32_t addr;
    float fCost;
} heap_t;

typedef struct {
    u_int32_t id;
    float gCost;
    float fCost;
    u_int32_t parent;
} node_t;

typedef struct {
    u_int32_t id;
    float gCost;
} sort_t;

#endif