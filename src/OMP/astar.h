#ifndef ASTAR_H
#define ASTAR_H

#define MAX_NODE_EXPAND 8
#define MAX_QUEUE_SIZE 1000000

typedef struct __node node;
typedef struct __neighbors_list neighbors_list;
typedef struct __path path;
typedef struct __list list;

struct __node {
    int id;
    double gCost;
    double fCost;
    node *parent;
};

struct __path {
    int count;
    int *nodeIds;
    double cost;
};

struct __neighbors_list {
    int capacity;
    int count;
    int *nodeIds;
    double *costs;
};

typedef struct {
    int max_size;
    void (*get_neighbors)(neighbors_list *neighbors, int n_id);
    double (*heuristic)(int n1_id, int n2_id);
} AStarSource;

path *find_path_omp(AStarSource *source, int s_id, int t_id, int k, double *time);

void add_neighbor(neighbors_list *neighbors, int n_id, double cost);

void path_destroy(path *path);

node *node_create(int id, double gCost, double fCost, node *parent);


#endif