#ifndef ASTAR_SEQ_H
#define ASTAR_SEQ_H

#include <stdlib.h>

#define DEFAULT_NIEGHBORS_LIST_CAPACITY 8;

typedef unsigned long astar_id_t;
typedef struct __node node;
typedef struct __neighbors_list neighbors_list;
typedef struct __path path;

struct __node {
    astar_id_t id;
    node *parent;
    double gCost;
    double fCost;
};

struct __path {
    size_t count;
    astar_id_t *nodeIds;
    double cost;
};

struct __neighbors_list {
    size_t capacity;
    size_t count;
    astar_id_t *nodeIds;
    double *costs;
};

typedef struct {
    size_t max_size;
    void (*get_neighbors)(neighbors_list *neighbors, astar_id_t n_id);
    double (*heuristic)(astar_id_t n1_id, astar_id_t n2_id);
} AStarSource;

path *find_path_sequential(AStarSource *source, astar_id_t s_id, astar_id_t t_id);

void add_neighbor(neighbors_list *neighbors, astar_id_t n_id, double cost);

void path_destroy(path *path);


#endif