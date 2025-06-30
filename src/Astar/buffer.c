#include <stdlib.h>
#include <string.h>
#include "buffer.h" 

buffer_t *buffer_init(void) {
    buffer_t *buffer = malloc(sizeof(buffer_t));
    buffer->size = 0;
    buffer->capacity = INIT_BUFFER_CAPACITY;
    buffer->elems = malloc(buffer->capacity * sizeof(buffer_elem_t));
    return buffer;
}

void buffer_destroy(buffer_t *buffer) {
    free(buffer->elems);
    free(buffer);
}

void buffer_insert(buffer_t *buffer, buffer_elem_t elem) {
    if (buffer->size == buffer->capacity) {
        buffer->capacity *= 2;
        buffer->elems = realloc(buffer->elems, buffer->capacity * sizeof(buffer_elem_t));
    }
    buffer->elems[buffer->size++] = elem;
}

void fill_buffer(buffer_t *to, buffer_t *from) {
    if (to->capacity < to->size + from->size) {
        to->capacity = to->capacity + from->size;
        to->elems = realloc(to->elems, to->capacity * sizeof(buffer_elem_t));
    }
    memcpy(to->elems + to->size, from->elems, from->size * sizeof(buffer_elem_t));
    to->size += from->size;
    from->size = 0;
}