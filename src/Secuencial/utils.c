#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "MapReader.h"
#include "astar.h"


// Heuristics

double ChevyshevHeuristic(astar_id_t n1_id, astar_id_t n2_id) {
    Node n1 = GetNodeById(MAP, n1_id);
    Node n2 = GetNodeById(MAP, n2_id);
    double distX = abs(n2->x - n1->x);
    double distY = abs(n2->y - n1->y);
    if (distX > distY) return distX;
    else return distY;
}

double MahattanHeuristic(astar_id_t n1_id, astar_id_t n2_id) {
    Node n1 = GetNodeById(MAP, n1_id);
    Node n2 = GetNodeById(MAP, n2_id);
    return abs(n2->x - n1->x) + abs(n2->y - n1->y);
}

double DiagonalHeuristic(astar_id_t n1_id, astar_id_t n2_id) {
    Node n1 = GetNodeById(MAP, n1_id);
    Node n2 = GetNodeById(MAP, n2_id);
    double dx = abs(n2->x - n1->x);
    double dy = abs(n2->y - n1->y);
    return dx + dy - fmin(dx, dy) * (sqrt(2) - 1); 
}



// Get Neighbors

void GetNeighbors8Tiles(neighbors_list *neighbors, astar_id_t n_id) {
    Node node = GetNodeById(MAP, n_id);
    for (int j = -1; j < 2; j++) {
        for (int i = -1; i < 2; i++) {
            if (i == 0 && j == 0) continue;
            if (ExistsNodeAtPos(MAP, node->x+i, node->y+j)) {
                add_neighbor(neighbors, GetIdAtPos(MAP, node->x+i, node->y+j), 1);
            }
        }
    }
}

void GetNeighbors4Tiles(neighbors_list *neighbors, astar_id_t n_id) {
    Node node = GetNodeById(MAP, n_id);
    if (ExistsNodeAtPos(MAP, node->x-1, node->y)) add_neighbor(neighbors, GetIdAtPos(MAP, node->x-1, node->y), 1);
    if (ExistsNodeAtPos(MAP, node->x+1, node->y)) add_neighbor(neighbors, GetIdAtPos(MAP, node->x+1, node->y), 1);
    if (ExistsNodeAtPos(MAP, node->x, node->y-1)) add_neighbor(neighbors, GetIdAtPos(MAP, node->x, node->y-1), 1);
    if (ExistsNodeAtPos(MAP, node->x, node->y+1)) add_neighbor(neighbors, GetIdAtPos(MAP, node->x, node->y+1), 1);
}

void GetNeighbors(neighbors_list *neighbors, astar_id_t n_id) {
    Node node = GetNodeById(MAP, n_id);

    // Nodo izquierda
    if (ExistsNodeAtPos(MAP, node->x-1, node->y)) add_neighbor(neighbors, GetIdAtPos(MAP, node->x-1, node->y), 1);
    // Nodo derecha
    if (ExistsNodeAtPos(MAP, node->x+1, node->y)) add_neighbor(neighbors, GetIdAtPos(MAP, node->x+1, node->y), 1);
    // Nodo arriba
    if (ExistsNodeAtPos(MAP, node->x, node->y-1)) add_neighbor(neighbors, GetIdAtPos(MAP, node->x, node->y-1), 1);
    // Nodo abajo
    if (ExistsNodeAtPos(MAP, node->x, node->y+1)) add_neighbor(neighbors, GetIdAtPos(MAP, node->x, node->y+1), 1);
    
    // Nodo arriba izquierda
    if (ExistsNodeAtPos(MAP, node->x-1, node->y) && ExistsNodeAtPos(MAP, node->x, node->y-1) && ExistsNodeAtPos(MAP, node->x-1, node->y-1)) {
        add_neighbor(neighbors, GetIdAtPos(MAP, node->x-1, node->y-1), sqrt(2));
    }
    // Nodo abajo izqueirda
    if (ExistsNodeAtPos(MAP, node->x-1, node->y) && ExistsNodeAtPos(MAP, node->x, node->y+1) && ExistsNodeAtPos(MAP, node->x-1, node->y+1)) {
        add_neighbor(neighbors, GetIdAtPos(MAP, node->x-1, node->y+1), sqrt(2));
    }
    // Nodo arriba derecha
    if (ExistsNodeAtPos(MAP, node->x+1, node->y) && ExistsNodeAtPos(MAP, node->x, node->y-1) && ExistsNodeAtPos(MAP, node->x+1, node->y-1)) {
        add_neighbor(neighbors, GetIdAtPos(MAP, node->x+1, node->y-1), sqrt(2));
    }
    // Nodo abajo derecha
    if (ExistsNodeAtPos(MAP, node->x+1, node->y) && ExistsNodeAtPos(MAP, node->x, node->y+1) && ExistsNodeAtPos(MAP, node->x+1, node->y+1)) {
        add_neighbor(neighbors, GetIdAtPos(MAP, node->x+1, node->y+1), sqrt(2));
    }
}



// Print path

void PrintPath(path *path) {
    printf("Path found!\n");
    printf("Number of nodes = %zu\n", path->count);
    printf("Total cost = %f\n", path->cost);
    for (int i = 0; i < path->count; i++) {
        printf("[%d,%d]\n", GetNodeById(MAP, path->nodeIds[i])->x, GetNodeById(MAP, path->nodeIds[i])->y);
    }
    printf("\n");
}