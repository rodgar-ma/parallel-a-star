#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "MapUtils.h"
#include "astar.h"

Map LoadMap(char *filename) {
    FILE *file = fopen(filename, "r");

    if (!file) {
        perror("Error abriendo el archivo");
        return NULL;
    }

    int width = 0, height = 0;
    char buffer[1024];

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

    // Reservar memoria para el mapa
    Map map = malloc(sizeof(struct __Map));
    map->width = width;
    map->height = height;
    map->grid = calloc(height, sizeof(Node **));

    // Leer el mapa
    for (int y = 0; y < height; y++) {
        fgets(buffer, sizeof(buffer), file);
        map->grid[y] = calloc(width, sizeof(Node *));
        for (int x = 0; x < width; x++) {
            if (buffer[x] == '.') {
                map->grid[y][x] = malloc(sizeof(struct __Node));
                map->grid[y][x]->x = x;
                map->grid[y][x]->y = y;
                map->grid[y][x]->id = y*map->width + x;
            } else {
                map->grid[y][x] = NULL;
            }
        }
    }

    // Cierra el fichero
    fclose(file);
    MAP = map;
    return map;
}

Node GetNodeById(Map map, astar_id_t id) {
    int y = id / map->width;
    int x = id - (y * map->width);
    if (map->grid[y][x] == NULL) return NULL;
    else return map->grid[y][x];
}

int ExistsNodeAtPos(Map map, int x, int y) {
    if (x < 0 || x >= map->width || y < 0 || y >= map->height) return 0;
    return map->grid[y][x] != NULL;
}

astar_id_t GetIdAtPos(Map map, int x, int y) {
    return map->grid[y][x]->id;
}

void FreeMap(Map map) {
    for (int y = 0; y < map->height; y++) {
        for (int x = 0; x < map->width; x++) {
            if (map->grid[y][x]) free(map->grid[y][x]);
        }
        free(map->grid[y]);
    }
    free(map->grid);
    free(map);
}

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
    if (!node) return;

    // Movimientos ortogonales (costo 1)
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};
    for (int i = 0; i < 4; i++) {
        if (ExistsNodeAtPos(MAP, node->x + dx[i], node->y + dy[i])) {
            add_neighbor(neighbors, GetIdAtPos(MAP, node->x + dx[i], node->y + dy[i]), 1);
        }
    }

    // Movimientos diagonales (costo sqrt(2))
    int ddx[] = {-1, -1, 1, 1};
    int ddy[] = {-1, 1, -1, 1};
    for (int i = 0; i < 4; i++) {
        if (ExistsNodeAtPos(MAP, node->x + ddx[i], node->y + ddy[i])) {
            add_neighbor(neighbors, GetIdAtPos(MAP, node->x + ddx[i], node->y + ddy[i]), sqrt(2));
        }
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