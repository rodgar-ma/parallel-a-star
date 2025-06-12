#ifndef ASTAR_GPU_H
#define ASTAR_GPU_H

typedef struct node_t {
    int id;
    float gCost;
    float fCost;
    int parent;
    unsigned int is_open:1;
    int open_index;
} node_t;

typedef struct neighbors_list {
    int capacity;
    int count;
    int *nodeIds;
    float *costs;
} neighbors_list;

typedef struct path {
    int count;
    int *nodeIds;
    float cost;
} path;

typedef struct {
    int max_size;
    void (*get_neighbors)(neighbors_list *neighbors, int n_id);
    float (*heuristic)(int n1_id, int n2_id);
} AStarSource;

path *find_path(AStarSource *source, int start_id, int target_id, double *time);

#endif