#ifndef ASTAR_H
#define ASTAR_H

#define INIT_NEIGHBORS_LIST_CAPACITY 10

typedef struct {
    int id;
    float gCost;
    float fCost;
    int parent;
    unsigned int is_open:1;
    int open_index;
} node_t;

typedef struct {
    int capacity;
    int count;
    int *nodeIds;
    float *costs;
} neighbors_list;

typedef struct {
    int count;
    int *nodeIds;
    float cost;
} path;

typedef struct {
    int max_size;
    void (*get_neighbors)(neighbors_list *neighbors, int n_id);
    float (*heuristic)(int n1_id, int n2_id);
} AStarSource;

node_t *node_create(int id, float gCost, float fCost, int parent);

neighbors_list *neighbors_list_create();

void neighbors_list_destroy(neighbors_list *list);

void add_neighbor(neighbors_list *neighbors, int n_id, float cost);

path *retrace_path(node_t ** visited, int target);

void path_destroy(path *p);

void visited_list_destroy(node_t **visited, int size);

path *astar_search(AStarSource *source, int s_id, int t_id, double *cpu_time_used);

void path_destroy(path *path);

#endif