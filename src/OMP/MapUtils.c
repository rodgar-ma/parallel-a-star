#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "MapUtils.h"
#include "astar.h"

// Carga el mapa del fichero `filename`.
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
        buffer[strcspn(buffer, "\r\n")] = 0;
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

// Devuelve el `Node` en `map` correspondiente al `id`.
Node GetNodeById(Map map, int id) {
    int y = id / map->width;
    int x = id % map->width;
    if (map->grid[y][x] == NULL) return NULL;
    else return map->grid[y][x];
}

// Devuelve `true` si hay un nodo en `map` con coordenadas `x` e `y`.
// Devuelve `false` en caso contrario.
int ExistsNodeAtPos(Map map, int x, int y) {
    if (x < 0 || x >= map->width || y < 0 || y >= map->height) return 0;
    return map->grid[y][x] != NULL;
}

// Devuelve el `int` del nodo en `map` con coordenadas `x` e `y`.
int GetIdAtPos(Map map, int x, int y) {
    return map->grid[y][x]->id;
}

// Libera `map`.
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

// Chevyshev Heuristic.
double ChevyshevHeuristic(int n1_id, int n2_id) {
    Node n1 = GetNodeById(MAP, n1_id);
    Node n2 = GetNodeById(MAP, n2_id);
    double distX = abs(n2->x - n1->x);
    double distY = abs(n2->y - n1->y);
    if (distX > distY) return distX;
    else return distY;
}

// Manhattan Heuristic.
double MahattanHeuristic(int n1_id, int n2_id) {
    Node n1 = GetNodeById(MAP, n1_id);
    Node n2 = GetNodeById(MAP, n2_id);
    return abs(n2->x - n1->x) + abs(n2->y - n1->y);
}

// Diagonal Heuristic
double DiagonalHeuristic(int n1_id, int n2_id) {
    Node n1 = GetNodeById(MAP, n1_id);
    Node n2 = GetNodeById(MAP, n2_id);
    double dx = abs(n2->x - n1->x);
    double dy = abs(n2->y - n1->y);
    return dx + dy + (sqrt(2) - 2.0) * fmin(dx, dy); 
}

// Euclidean Heuristic
double EuclideanHeuristic(int n1_id, int n2_id) {
    Node n1 = GetNodeById(MAP, n1_id);
    Node n2 = GetNodeById(MAP, n2_id);
    double dx = abs(n2->x - n1->x);
    double dy = abs(n2->y - n1->y);
    return sqrt(dx * dx + dy * dy); 
}



// Get Neighbors
void GetNeighbors(neighbors_list *neighbors, int n_id) {
    Node node = GetNodeById(MAP, n_id);

    // Movimientos ortogonales (coste 1)
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};
    for (int i = 0; i < 4; i++) {
        if (ExistsNodeAtPos(MAP, node->x + dx[i], node->y + dy[i])) {
            add_neighbor(neighbors, GetIdAtPos(MAP, node->x + dx[i], node->y + dy[i]), 1);
        }
    }

    // Movimientos diagonales (coste sqrt(2))
    if (ExistsNodeAtPos(MAP, node->x - 1, node->y) && ExistsNodeAtPos(MAP, node->x, node->y - 1)
    && ExistsNodeAtPos(MAP, node->x - 1, node->y - 1)) add_neighbor(neighbors, GetIdAtPos(MAP, node->x - 1, node->y - 1), sqrt(2));
    
    if (ExistsNodeAtPos(MAP, node->x - 1, node->y) && ExistsNodeAtPos(MAP, node->x, node->y + 1)
    && ExistsNodeAtPos(MAP, node->x - 1, node->y + 1)) add_neighbor(neighbors, GetIdAtPos(MAP, node->x - 1, node->y + 1), sqrt(2));
    
    if (ExistsNodeAtPos(MAP, node->x + 1, node->y) && ExistsNodeAtPos(MAP, node->x, node->y - 1)
    && ExistsNodeAtPos(MAP, node->x + 1, node->y - 1)) add_neighbor(neighbors, GetIdAtPos(MAP, node->x + 1, node->y - 1), sqrt(2));
    
    if (ExistsNodeAtPos(MAP, node->x + 1, node->y) && ExistsNodeAtPos(MAP, node->x, node->y + 1)
    && ExistsNodeAtPos(MAP, node->x + 1, node->y + 1)) add_neighbor(neighbors, GetIdAtPos(MAP, node->x + 1, node->y + 1), sqrt(2));
}



// Print path
void PrintPath(path *path) {
    printf("Path found!\n");
    printf("Number of nodes = %d\n", path->count);
    printf("Total cost = %f\n", path->cost);
    for (int i = 0; i < path->count; i++) {
        printf("[%d,%d]\n", GetNodeById(MAP, path->nodeIds[i])->x, GetNodeById(MAP, path->nodeIds[i])->y);
    }
    printf("\n");
}