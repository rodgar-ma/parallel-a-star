#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "AStar.h"

struct __Node {
    int x;
    int y;
};

struct __Map {
    int width;
    int height;
    Node ** grid;
};

typedef struct __Node *Node;
typedef struct __Map *Map;

Map map;

Map *LoadMap(const char *filename) {
    FILE *file;
    int err;

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
    Map *map = malloc(sizeof(Map));
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

void FreeMap(Map *map) {
    for (int y = 0; y < map->height; y++) {
        for (int x = 0; x < map->width; x++) {
            if (map->grid[y][x]) free(map->grid[y][x]);
        }
        free(map->grid[y]);
    }
    free(map->grid);
    free(map);
}

void nodeNeighbors(ASNeighborList neighbors, void *node, void *context) {
    for (int y = 0; y < map->height; y++) {
        for (int x = 0; x < map->width; x++) {
            if (map->grid[y][x]) {
                if (x > 0 && map->grid[y][x - 1]) ASNeighborListAdd(neighbors, map->grid[y][x - 1], 1);
                if (x < map->width - 1 && map->grid[y][x + 1]) ASNeighborListAdd(neighbors, map->grid[y][x + 1], 1);
                if (y > 0 && map->grid[y - 1][x]) ASNeighborListAdd(neighbors, map->grid[y - 1][x], 1);
                if (y < map->height - 1 && map->grid[y + 1][x]) ASNeighborListAdd(neighbors, map->grid[y + 1][x], 1);
            }
        }
    }
}

float pathCostHeuristic(void *fromNode, void *toNode, void *context) {
    return abs(((Node)toNode)->x - ((Node)fromNode)->x) + abs(((Node)toNode)->y - ((Node)fromNode)->y);
}

int earlyExit(size_t visitedCount, void *visitingNode, void *goalNode, void *context) {
    return 0;
}

int nodeComparator(void *node1, void *node2, void *context) {
    int distX = ((Node)node1)->x - ((Node)node2)->x;
    int distY = ((Node)node1)->y - ((Node)node2)->y;
    if (distX > 0) {
        if (distY > 0) return 1;
        else return 1;
    } else {
        if (distY > 0) return 1;
        else return -1;
    }
    return 0;
}

/***********************************************************************/

int main(int argc, char const *argv[])
{
    printf("\n");
    char* filename = "../Astar/maze-map/maze512-1-0.map";
    map = LoadMap(filename);

    ASPathNodeSource source = {.nodeSize = sizeof(Node),
                               .nodeNeighbors = &nodeNeighbors,
                               .pathCostHeuristic = &pathCostHeuristic,
                               .earlyExit = &earlyExit,
                               .nodeComparator = &nodeComparator};
    
    ASPathCreate(&source, map->grid[1][1], map->grid[511][511], NULL);

    return 0;
}
