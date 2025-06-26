#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "MapUtils.h"


// Carga el mapa del fichero `filename`.
Map *LoadMap(char *filename) {

    // printf("Abriendo: %s\n", filename);
    FILE *file = fopen(filename, "r");

    if (!file) {
        perror("Error abriendo el archivo");
        return NULL;
    }

    int width = 0, height = 0;
    char buffer[64];

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
    Map *map = malloc(sizeof(Map));
    map->width = width;
    map->height = height;
    map->grid = malloc(height * sizeof(Node**));

    char *line_buffer = malloc(width + 3);

    // Leer el mapa
    for (int y = 0; y < height; y++) {
        fgets(line_buffer, width + 3, file);
        map->grid[y] = malloc(width * sizeof(Node*));
        for (int x = 0; x < width; x++) {
            map->grid[y][x] = malloc(sizeof(Node));
            map->grid[y][x]->x = x;
            map->grid[y][x]->y = y;
            map->grid[y][x]->id = y * map->width + x;
            if (line_buffer[x] == '.') {
                map->grid[y][x]->walkable = 1;
            } else {
                map->grid[y][x]->walkable = 0;
            }
        }
    }
    free(line_buffer);

    // Cierra el fichero
    fclose(file);
    return map;
}

// Libera `map`.
void FreeMap(Map *map) {
    for (int y = 0; y < map->height; y++) {
        free(map->grid[y]);
    }
    free(map->grid);
    free(map);
}

// Devuelve el `Node` en `map` correspondiente al `id`.
void idToXY(int id, int *x, int *y) {
    *x = id % MAP->width;
    *y = id / MAP->width;
}

int xyToID(int x, int y) {
    return y * MAP->width + x;
}

int inrange(int x, int y) {
    return x >= 0 && y >= 0 && x < MAP->width && y < MAP->height;
}

int is_walkable(int x, int y) {
    if (inrange(x, y)) {
        return MAP->grid[y][x]->walkable;
    }
    return 0;
}


// Chevyshev Heuristic.
float ChevyshevHeuristic(int n1_id, int n2_id) {
    int n1_x, n1_y, n2_x, n2_y;
    idToXY(n1_id, &n1_x, &n1_y);
    idToXY(n2_id, &n2_x, &n2_y);
    int error = 0;
    if (!inrange(n1_x, n1_y)) {
        printf("Error: Nodo (%d, %d) fuera del mapa.\n", n1_x, n1_y);
        error = 1;
    }
    if (!inrange(n2_x, n2_y)) {
        printf("Error: Nodo (%d, %d) fuera del mapa.\n", n2_x, n2_y);
        error = 1;
    }
    if (error) return FLT_MAX;
    
    float distX = abs(n2_x - n1_x);
    float distY = abs(n2_y - n1_y);
    if (distX > distY) return distX;
    else return distY;
}

// Manhattan Heuristic.
float ManhattanHeuristic(int n1_id, int n2_id) {
    int n1_x, n1_y, n2_x, n2_y;
    idToXY(n1_id, &n1_x, &n1_y);
    idToXY(n2_id, &n2_x, &n2_y);
    int error = 0;
    if (!inrange(n1_x, n1_y)) {
        printf("Error: Nodo (%d, %d) fuera del mapa.\n", n1_x, n1_y);
        error = 1;
    }
    if (!inrange(n2_x, n2_y)) {
        printf("Error: Nodo (%d, %d) fuera del mapa.\n", n2_x, n2_y);
        error = 1;
    }
    if (error) return FLT_MAX;

    return abs(n2_x - n1_x) + abs(n2_y - n1_y);
}

// Diagonal Heuristic
float DiagonalHeuristic(int n1_id, int n2_id) {
    int n1_x, n1_y, n2_x, n2_y;
    idToXY(n1_id, &n1_x, &n1_y);
    idToXY(n2_id, &n2_x, &n2_y);
    int error = 0;
    if (!inrange(n1_x, n1_y)) {
        printf("Error: Nodo (%d, %d) fuera del mapa.\n", n1_x, n1_y);
        error = 1;
    }
    if (!inrange(n2_x, n2_y)) {
        printf("Error: Nodo (%d, %d) fuera del mapa.\n", n2_x, n2_y);
        error = 1;
    }
    if (error) return FLT_MAX;

    float dx = abs(n2_x - n1_x);
    float dy = abs(n2_y - n1_y);
    return dx + dy + (sqrt(2) - 2.0) * fmin(dx, dy); 
}

// Euclidean Heuristic
float EuclideanHeuristic(int n1_id, int n2_id) {
    int n1_x, n1_y, n2_x, n2_y;
    idToXY(n1_id, &n1_x, &n1_y);
    idToXY(n2_id, &n2_x, &n2_y);
    int error = 0;
    if (!inrange(n1_x, n1_y)) {
        printf("Error: Nodo (%d, %d) fuera del mapa.\n", n1_x, n1_y);
        error = 1;
    }
    if (!inrange(n2_x, n2_y)) {
        printf("Error: Nodo (%d, %d) fuera del mapa.\n", n2_x, n2_y);
        error = 1;
    }
    if (error) return FLT_MAX;

    float dx = abs(n2_x - n1_x);
    float dy = abs(n2_y - n1_y);
    return sqrt(dx * dx + dy * dy); 
}



// Get Neighbors
void GetNeighbors(neighbors_list *neighbors, int n_id) {
    int x, y;
    idToXY(n_id, &x, &y);
    if (!inrange(x, y)) {
        printf("Error: Nodo (%d, %d) fuera del mapa.\n", x, y);
        return;
    }

    // Movimientos ortogonales (coste 1)
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};
    for (int i = 0; i < 4; i++) {
        if (is_walkable(x + dx[i], y + dy[i])) {
            add_neighbor(neighbors, xyToID(x + dx[i], y + dy[i]), 1);
        }
    }

    // Movimientos diagonales (coste sqrt(2))
    if (is_walkable(x - 1, y) && is_walkable(x, y - 1) && is_walkable(x - 1, y - 1))
        add_neighbor(neighbors, xyToID(x - 1, y - 1), sqrt(2));
    
    if (is_walkable(x - 1, y) && is_walkable(x, y + 1) && is_walkable(x - 1, y + 1))
        add_neighbor(neighbors, xyToID(x - 1, y + 1), sqrt(2));

    if (is_walkable(x + 1, y) && is_walkable(x, y - 1) && is_walkable(x + 1, y - 1))
        add_neighbor(neighbors, xyToID(x + 1, y - 1), sqrt(2));

    if (is_walkable(x + 1, y) && is_walkable(x, y + 1) && is_walkable(x + 1, y + 1))
        add_neighbor(neighbors, xyToID(x + 1, y + 1), sqrt(2));
}

// Get Four Neighbors
void GetFourNeighbors(neighbors_list *neighbors, int n_id) {
    int x, y;
    idToXY(n_id, &x, &y);
    if (!inrange(x, y)) {
        printf("Error: Nodo (%d, %d) fuera del mapa.\n", x, y);
        return;
    }

    // Movimientos ortogonales (coste 1)
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};
    for (int i = 0; i < 4; i++) {
        if (is_walkable(x + dx[i], y + dy[i])) {
            add_neighbor(neighbors, xyToID(x, y), 1);
        }
    }
}

// Print path
void PrintPath(path *path) {
    printf("Path found!\n");
    printf("Number of nodes = %u\n", path->count);
    printf("Total cost = %f\n", path->cost);
    int x, y;
    for (int i = 0; i < path->count; i++) {
        idToXY(path->nodeIds[i], &x, &y);
        printf("[%d,%d]\n", x, y);
    }
    printf("\n");
}