#ifndef ASTAR_SEQ_H
#define ASTAR_SEQ_H

#include <stdlib.h>

#define DEFAULT_NIEGHBORS_LIST_CAPACITY 8

typedef struct __node node;
typedef struct __neighbors_list neighbors_list;
typedef struct __path path;

struct __node {
    int id;
    node *parent;
    double gCost;
    double fCost;
    unsigned int isOpen:1;
    size_t open_index;
};

struct __path {
    size_t count;
    int *nodeIds;
    double cost;
};

struct __neighbors_list {
    size_t capacity;
    size_t count;
    int *nodeIds;
    double *costs;
};

typedef struct {
    size_t max_size;
    void (*get_neighbors)(neighbors_list *neighbors, int n_id);
    double (*heuristic)(int n1_id, int n2_id);
} AStarSource;

path *find_path_sequential(AStarSource *source, int s_id, int t_id, double *cpu_time_used);

void add_neighbor(neighbors_list *neighbors, int n_id, double cost);

void path_destroy(path *path);

node *node_create(int id, double gCost, double fCost, node *parent);

#endif