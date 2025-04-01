#include <stdlib.h>
#include "AStar.h"

typedef struct {
    int x, y;
} GridNode;

float grid_heuristic(Node a, Node b) {
    GridNode* ga = (GridNode*)a;
    GridNode* gb = (GridNode*)b;
    // Manhattan distance
    return abs(ga->x - gb->x) + abs(ga->y - gb->y);
}

Node* grid_neighbors(Node current, int* count) {
    GridNode* gc = (GridNode*)current;
    *count = 4;
    GridNode* neighbors = malloc(sizeof(GridNode) * 4);
    
    neighbors[0] = (GridNode){gc->x + 1, gc->y};
    neighbors[1] = (GridNode){gc->x - 1, gc->y};
    neighbors[2] = (GridNode){gc->x, gc->y + 1};
    neighbors[3] = (GridNode){gc->x, gc->y - 1};
    
    return (Node*)neighbors;
}

float grid_cost(Node from, Node to) {
    // Uniform cost in this simple grid
    return 1.0f;
}

int grid_compare(Node a, Node b) {
    GridNode* ga = (GridNode*)a;
    GridNode* gb = (GridNode*)b;
    return ga->x == gb->x && ga->y == gb->y;
}

int main() {
    GridNode start = {0, 0};
    GridNode goal = {4, 4};
    
    int path_length = 0;
    Node* path = a_star(&start, &goal, 
                        grid_heuristic, 
                        grid_neighbors, 
                        grid_cost, 
                        grid_compare, 
                        &path_length);
    
    if (path) {
        printf("Path found with %d steps:\n", path_length);
        for (int i = 0; i < path_length; i++) {
            GridNode* node = (GridNode*)path[i];
            printf("(%d, %d)\n", node->x, node->y);
        }
        free(path);
    } else {
        printf("No path found.\n");
    }
    
    return 0;
}