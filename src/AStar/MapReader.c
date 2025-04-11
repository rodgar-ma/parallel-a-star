#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "MapReader.h"

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
    map->count = 0;

    // Leer el mapa
    for (int y = 0; y < height; y++) {
        fgets(buffer, sizeof(buffer), file);
        map->grid[y] = calloc(width, sizeof(Node *));

        for (int x = 0; x < width; x++) {
            if (buffer[x] == '.') {
                map->grid[y][x] = malloc(sizeof(struct __Node));
                map->grid[y][x]->x = x;
                map->grid[y][x]->y = y;
                map->grid[y][x]->id = map->count++;
            } else {
                map->grid[y][x] = NULL;
                map->count++;
            }
        }
    }

    fclose(file);

    return map;
}

Node GetNodeById(Map map, id_t id) {
    int y = id / map->width;
    int x = id - (y * map->width);
    if (map->grid[y][x] == NULL) return NULL;
    else return map->grid[y][x];
}

int ExistsNodeAtPos(Map map, int x, int y) {
    if (x < 0 || x >= map->width || y < 0 || y >= map->height) return 0;
    return map->grid[y][x] != NULL;
}

id_t GetIdAtPos(Map map, int x, int y) {
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