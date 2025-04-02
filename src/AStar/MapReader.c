#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "MapReader.h"

Map LoadMap(const char *filename) {
    FILE *file;
    errno_t err;

    if ((err = fopen_s(&file, filename, "r")) != 0) {
        perror("Error abriendo el archivo");
        return NULL;
    }

    int width = 0, height = 0;
    char buffer[1024];

    // Leer dimensiones
    while (fgets(buffer, sizeof(buffer), file)) {
        if (strncmp(buffer, "width", 5) == 0) {
            if ((err = sscanf_s(buffer, "width %d", &width)) != 1) {
                perror("Error abriendo el archivo");
                return NULL;
            }
        } else if (strncmp(buffer, "height", 6) == 0) {
            if ((err = sscanf_s(buffer, "height %d", &height)) != 1) {
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
            } else {
                map->grid[y][x] = NULL;
            }
        }
    }

    fclose(file);

    return map;
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