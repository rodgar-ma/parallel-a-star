#include <stdio.h>
#include "graph.h"

#define INIT_NEIGHBORS_SIZE 8
#define SQRT2 1.4142135624

Graph *create_graph(int num_nodes) {
    Graph *g = malloc(sizeof(Graph));
    g->num_nodes = num_nodes;
    g->nodes = calloc(num_nodes, sizeof(Node *));
    return g;
}

Node *create_node(int id) {
    Node *n = malloc(sizeof(Node));
    n->id = id;
    n->neighbors_list = malloc(sizeof(Neighbors));
    n->neighbors_list->size = 0;
    n->neighbors_list->capacity = INIT_NEIGHBORS_SIZE;
    n->neighbors_list->neighbors = calloc(INIT_NEIGHBORS_SIZE, sizeof(Node *));
    n->neighbors_list->costs = calloc(INIT_NEIGHBORS_SIZE, sizeof(float));
    return n;
}

void add_node(Graph *graph, int id) {
    if (id < 0 || id >= graph->num_nodes) return;
    if (graph->nodes[id] != NULL) return;
    graph->nodes[id] = create_node(id);
}

void add_edge(Graph *graph, int src, int dest, float cost) {
    if (src < 0 || src >= graph->num_nodes || dest < 0 || dest >= graph->num_nodes) return;
    Neighbors *neighbors_list = graph->nodes[src]->neighbors_list;
    if (neighbors_list->size == neighbors_list->capacity) {
        neighbors_list->capacity *= 2;
        neighbors_list->neighbors = realloc(neighbors_list->neighbors, neighbors_list->capacity * sizeof(int));
        neighbors_list->costs = realloc(neighbors_list->costs, neighbors_list->capacity * sizeof(float));
    }
    neighbors_list->neighbors[neighbors_list->size] = graph->nodes[dest];
    neighbors_list->costs[neighbors_list->size] = cost;
    neighbors_list->size++;
}

Neighbors *get_neighbors(Node *n) {
    return n->neighbors_list;
}

void delete_node(Node *n) {
    free(n->neighbors_list->neighbors);
    free(n->neighbors_list->costs);
    free(n->neighbors_list);
    free(n);
}

void delete_graph(Graph *graph) {
    for (int i = 0; i < graph->num_nodes; i++) {
        if (graph->nodes[i] != NULL) {
            delete_node(graph->nodes[i]);
        }
    }
    free(graph->nodes);
    free(graph);
}

Graph *create_graph_from_file(char *filename) {
    FILE *file = fopen(filename, "r");

    if (!file) {
        perror("Error abriendo el archivo");
        return NULL;
    }

    int width = 0, height = 0;
    char buffer[1024];

    int **grid = malloc(width * sizeof(int));
    for (int i = 0; i < width; i++) {
        grid[i] = malloc(height * sizeof(int));
    }

    // Leer dimensiones
    while (fgets(buffer, sizeof(buffer), file)) {
        if (strncmp(buffer, "width", 5) == 0) {
            if (sscanf(buffer, "width %d", &width) != 1) {
                perror("Error abriendo el archivo");
                return NULL;
            }
        } else if (strncmp(buffer, "height", 6) == 0) {
            if (sscanf(buffer, "height %d", &height) != 1) {
                perror("Error abriendo el archivo");
                return NULL;
            }
        } else if (strncmp(buffer, "map", 3) == 0) {
            break;  // Inicia la lectura del mapa
        }
    }

    if (width == 0 || height == 0) {
        printf("Error: dimensiones no especificadas correctamente.\n");
        fclose(file);
        return NULL;
    }

    // Leer el mapa
    for (int y = 0; y < height; y++) {
        fgets(buffer, sizeof(buffer), file);
        buffer[strcspn(buffer, "\r\n")] = 0;
        for (int x = 0; x < width; x++) {
            if (buffer[x] == '.') {
                grid[x][y] = 1;
            } else {
                grid[x][y] = 0;
            }
        }
    }
    
    // Cierra el fichero
    fclose(file);
    
    Graph *graph = create_graph(height * width);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (grid[x][y] == 1) {
                int id = y * width + x;
                int n_id;
                add_node(graph, id);
                if (x - 1 >= 0 && grid[x - 1][y]) {
                    n_id = y * width + x - 1;
                    add_edge(graph, id, n_id, 1.0);
                }
                if (x + 1 < width && grid[x + 1][y]) {
                    n_id = y * width + x + 1;
                    add_edge(graph, id, n_id, 1.0);
                }
                if (y - 1 >= 0 && grid[x][y - 1]) {
                    n_id = (y - 1) * width + x;
                    add_edge(graph, id, n_id, 1.0);
                }
                if (y + 1 < height && grid[x][y + 1]) {
                    n_id = (y + 1) * width + x;
                    add_edge(graph, id, n_id, 1.0);
                }
                if (x - 1 >= 0 && y - 1 >= 0 && grid[x - 1][y - 1] && grid[x - 1][y] && grid[x][y - 1]) {
                    n_id = (y - 1) * width + x - 1;
                    add_edge(graph, id, n_id, SQRT2);
                }
                if (x + 1 < width && y - 1 >= 0 && grid[x + 1][y - 1] && grid[x + 1][y] && grid[x][y - 1]) {
                    n_id = (y - 1) * width + x + 1;
                    add_edge(graph, id, n_id, SQRT2);
                }
                if (x - 1 >= 0 && y + 1 < height && grid[x - 1][y + 1] && grid[x - 1][y] && grid[x][y + 1]) {
                    n_id = (y + 1) * width + x - 1;
                    add_edge(graph, id, n_id, SQRT2);
                }
                if (x + 1 < width && y + 1 < height && grid[x + 1][y + 1] && grid[x + 1][y] && grid[x][y + 1]) {
                    n_id = (y + 1) * width + x + 1;
                    add_edge(graph, id, n_id, SQRT2);
                }
            }
        }
    }

    return graph;
}



