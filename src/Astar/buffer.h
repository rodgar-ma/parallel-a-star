#ifndef BUFFER_H
#define BUFFER_H

#define INIT_BUFFER_CAPACITY 10

typedef struct {
    int node_id;
    float gCost;
    int parent_id;
} buffer_elem_t;

typedef struct {
    int size;
    int capacity;
    buffer_elem_t *elems;
} buffer_t;

buffer_t *buffer_init(void);

void buffer_destroy(buffer_t *buffer);

void buffer_insert(buffer_t *buffer, buffer_elem_t elem);

void fill_buffer(buffer_t *to, buffer_t *from);

#endif