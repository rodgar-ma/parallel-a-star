#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "AStar.h"
#include "MapReader.h"

Map MAP;

double ChevyshevHeuristic(void *fromNode, void *toNode) {
    double distX = abs(((Node)toNode)->x - ((Node)fromNode)->x);
    double distY = abs(((Node)toNode)->y - ((Node)fromNode)->y);
    if (distX > distY) return distX;
    else return distY;
}

double MahattanHeuristic(void *fromNode, void *toNode) {
    return abs(((Node)toNode)->x - ((Node)fromNode)->x) + abs(((Node)toNode)->y - ((Node)fromNode)->y);
}

double DiagonalHeuristic(void *fromNode, void *toNode) {
    double dx = abs(((Node)toNode)->x - ((Node)fromNode)->x);
    double dy = abs(((Node)toNode)->y - ((Node)fromNode)->y);
    return dx + dy - fmin(dx, dy) * 0.4142135623730951; 
}

void GetNeighbors8Tiles(NeighborsList neighbors, void *node) {
    int x = ((Node)node)->x;
    int y = ((Node)node)->y;
    for (int j = -1; j < 2; j++) {
        for (int i = -1; i < 2; i++) {
            if (i == 0 && j == 0) continue;
            if (y+j > -1 && y+j < MAP->height && x+i > -1 && x+i < MAP->width && MAP->grid[y+j][x+i]) AddNeighbor(neighbors, (void*)MAP->grid[y+j][x+i], 1);
        }
    }
}

void GetNeighbors4Tiles(NeighborsList neighbors, void *node) {
    int x = ((Node)node)->x;
    int y = ((Node)node)->y;
    if (y-1 > -1 && MAP->grid[y-1][x]) AddNeighbor(neighbors, (void*)MAP->grid[y-1][x], 1);
    if (y+1 < MAP->height && MAP->grid[y+1][x]) AddNeighbor(neighbors, (void*)MAP->grid[y+1][x], 1);
    if (x-1 > -1 && MAP->grid[y][x-1]) AddNeighbor(neighbors, (void*)MAP->grid[y][x-1], 1);
    if (x+1 < MAP->width && MAP->grid[y][x+1]) AddNeighbor(neighbors, (void*)MAP->grid[y][x+1], 1);
}

void GetNeighbors(NeighborsList neighbors, void *node) {
    int x = ((Node)node)->x;
    int y = ((Node)node)->y;
    if (y-1 > -1 && MAP->grid[y-1][x]) AddNeighbor(neighbors, (void*)MAP->grid[y-1][x], 1);
    if (y+1 < MAP->height && MAP->grid[y+1][x]) AddNeighbor(neighbors, (void*)MAP->grid[y+1][x], 1);
    if (x-1 > -1 && MAP->grid[y][x-1]) AddNeighbor(neighbors, (void*)MAP->grid[y][x-1], 1);
    if (x+1 < MAP->width && MAP->grid[y][x+1]) AddNeighbor(neighbors, (void*)MAP->grid[y][x+1], 1);
    
    if (y-1 > -1 && x-1 > -1 && MAP->grid[y-1][x] && MAP->grid[y][x-1] && MAP->grid[y-1][x-1]) AddNeighbor(neighbors, (void*)MAP->grid[y-1][x-1], 1.4142135623730951);
    if (y-1 > -1 && x+1 < MAP->width && MAP->grid[y-1][x] && MAP->grid[y][x+1] && MAP->grid[y-1][x+1]) AddNeighbor(neighbors, (void*)MAP->grid[y-1][x+1], 1.4142135623730951);
    if (y+1 < MAP->height && x-1 > -1 && MAP->grid[y+1][x] && MAP->grid[y][x-1] && MAP->grid[y+1][x-1]) AddNeighbor(neighbors, (void*)MAP->grid[y+1][x-1], 1.4142135623730951);
    if (y+1 < MAP->height && x+1 < MAP->width && MAP->grid[y+1][x] && MAP->grid[y][x+1] && MAP->grid[y+1][x+1]) AddNeighbor(neighbors, (void*)MAP->grid[y+1][x+1], 1.4142135623730951);
}

int Equals(void *a, void *b) {
    Node na = (Node)a;
    Node nb = (Node)b;
    return na->x == nb->x && na->y == nb->y;
}

void PrintPath(Path path) {
    printf("Path found!\n");
    printf("Number of nodes = %zu\n", path->size);
    printf("Total cost = %f\n", path->cost);
    for (int i = 0; i < path->size; i++) {
        printf("[%d,%d]\n", ((Node)path->nodes[i])->x, ((Node)path->nodes[i])->y);
    }
    printf("\n");
}

typedef struct {
    int id;
    char filename[256];
    int width, height;
    int start_x, start_y;
    int goal_x, goal_y;
    double cost;
} MapScen;

void printMapScen(MapScen entry) {
        printf("Mapa: %s\n", entry.filename);
        printf("ID: %d\n", entry.id);
        printf("Archivo: %s\n", entry.filename);
        printf("Dimensiones: %dx%d\n", entry.width, entry.height);
        printf("Inicio: (%d, %d)\n", entry.start_x, entry.start_y);
        printf("Meta: (%d, %d)\n", entry.goal_x, entry.goal_y);
        printf("Coste: %.8f\n", entry.cost);
}

int main(int argc, char const *argv[])
{
    if (argc != 2) {
        perror("Uso: <exe> map.scen");
        return 1;
    }

    FILE *file = fopen(argv[1], "r");
    if (!file) {
        perror("No se pudo abrir el archivo");
        return 1;
    }

    AStarSource *source = malloc(sizeof(AStarSource));
    source->Heuristic = &DiagonalHeuristic;
    source->GetNeighbors = &GetNeighbors;
    source->Equals = &Equals;

    char map_file[256];
    char line[256];
    MapScen entry;
    int total_succeed = 0;
    int total_failed = 0;
    fgets(line, 256, file);
    while (fscanf(file, "%d\t%255s\t%d\t%d\t%d\t%d\t%d\t%d\t%lf\n",
                  &entry.id, entry.filename, &entry.width, &entry.height,
                  &entry.start_x, &entry.start_y, &entry.goal_x, &entry.goal_y,
                  &entry.cost) == 9)
    {
        if (!map_file || strcmp(map_file, entry.filename) != 0) {
            strcpy(map_file, entry.filename);
        }
        char newPath[256] = "../maze-map/";
        MAP = LoadMap(strcat(newPath, map_file));
        if (!MAP) {
            perror("Error al cargar el mapa");
            return 1;
        }

        source->map_size = MAP->count;

        Node start = GetNodeAtPos(MAP, entry.start_x, entry.start_y);
        Node goal = GetNodeAtPos(MAP, entry.goal_x, entry.goal_y);

        if (!start) {
            perror("No start node");
            return 1;
        } else if (!goal) {
            perror("No goal node");
            return 1;
        }

        Path path = FindPath(source, (void *)start, (void *)goal);

        printf("%d-", total_succeed + total_failed + 1);
        if (fabs(path->cost - entry.cost) < 0.0001) {
            printf("[OK] Coste esperado: %.8f, Coste encontrado: %.8f\n", entry.cost, path->cost);
            total_succeed++;
        }
        else {
            printf("[Error] Coste esperado: %.8f, Coste encontrado: %.8f\n", entry.cost, path->cost);
            total_failed++;
        }
        FreePath(path);
        FreeMap(MAP);
    }
    
    free(source);
    printf("\nResultados:\n");
    printf("Total de mapas: %d\n", total_succeed + total_failed);
    printf("Total de exitos: %d\n", total_succeed);
    printf("Total de fallos: %d\n", total_failed);
    return 0;
}
