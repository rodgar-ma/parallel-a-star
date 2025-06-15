#ifndef GRAPH_H
#define GRAPH_H

#include <stdlib.h>

typedef struct {
    int id;
    Neighbors *neighbors_list;
} Node;

typedef struct {
    int size;
    int capacity;
    Node **neighbors;
    float *costs;
} Neighbors;

typedef struct {
    int num_nodes;
    Node **nodes;
} Graph;

typedef struct {
    int height;
    int width;
    Graph *graph;
} GridGraph;

Graph *create_graph(int num_nodes);

Graph *create_graph_from_file(char *filename);

void add_node(Graph *graph, int id);

void add_edge(Graph *graph, int node1, int node2, float cost);

void delete_graph(Graph *graph);

#endif