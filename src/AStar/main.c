#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <math.h>
#include <time.h>
#include "astar.h"
#include "MapReader.h"

Map MAP;

double ChevyshevHeuristic(id_t n1_id, id_t n2_id) {
    Node n1 = GetNodeById(MAP, n1_id);
    Node n2 = GetNodeById(MAP, n2_id);
    double distX = abs(n2->x - n1->x);
    double distY = abs(n2->y - n1->y);
    if (distX > distY) return distX;
    else return distY;
}

double MahattanHeuristic(id_t n1_id, id_t n2_id) {
    Node n1 = GetNodeById(MAP, n1_id);
    Node n2 = GetNodeById(MAP, n2_id);
    return abs(n2->x - n1->x) + abs(n2->y - n1->y);
}

double DiagonalHeuristic(id_t n1_id, id_t n2_id) {
    Node n1 = GetNodeById(MAP, n1_id);
    Node n2 = GetNodeById(MAP, n2_id);
    double dx = abs(n2->x - n1->x);
    double dy = abs(n2->y - n1->y);
    return dx + dy - fmin(dx, dy) * (sqrt(2) - 1); 
}

void GetNeighbors8Tiles(neighbors_list *neighbors, id_t n_id) {
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

void GetNeighbors4Tiles(neighbors_list *neighbors, id_t n_id) {
    Node node = GetNodeById(MAP, n_id);
    if (ExistsNodeAtPos(MAP, node->x-1, node->y)) add_neighbor(neighbors, GetIdAtPos(MAP, node->x-1, node->y), 1);
    if (ExistsNodeAtPos(MAP, node->x+1, node->y)) add_neighbor(neighbors, GetIdAtPos(MAP, node->x+1, node->y), 1);
    if (ExistsNodeAtPos(MAP, node->x, node->y-1)) add_neighbor(neighbors, GetIdAtPos(MAP, node->x, node->y-1), 1);
    if (ExistsNodeAtPos(MAP, node->x, node->y+1)) add_neighbor(neighbors, GetIdAtPos(MAP, node->x, node->y+1), 1);
}

void GetNeighbors(neighbors_list *neighbors, id_t n_id) {
    Node node = GetNodeById(MAP, n_id);
    if (ExistsNodeAtPos(MAP, node->x-1, node->y)) add_neighbor(neighbors, GetIdAtPos(MAP, node->x-1, node->y), 1);
    if (ExistsNodeAtPos(MAP, node->x+1, node->y)) add_neighbor(neighbors, GetIdAtPos(MAP, node->x+1, node->y), 1);
    if (ExistsNodeAtPos(MAP, node->x, node->y-1)) add_neighbor(neighbors, GetIdAtPos(MAP, node->x, node->y-1), 1);
    if (ExistsNodeAtPos(MAP, node->x, node->y+1)) add_neighbor(neighbors, GetIdAtPos(MAP, node->x, node->y+1), 1);
    
    if (ExistsNodeAtPos(MAP, node->x-1, node->y) && ExistsNodeAtPos(MAP, node->x, node->y-1) && ExistsNodeAtPos(MAP, node->x-1, node->y-1)) {
        add_neighbor(neighbors, GetIdAtPos(MAP, node->x-1, node->y-1), sqrt(2));
    }
    if (ExistsNodeAtPos(MAP, node->x-1, node->y) && ExistsNodeAtPos(MAP, node->x, node->y+1) && ExistsNodeAtPos(MAP, node->x-1, node->y+1)) {
        add_neighbor(neighbors, GetIdAtPos(MAP, node->x-1, node->y+1), sqrt(2));
    }
    if (ExistsNodeAtPos(MAP, node->x+1, node->y) && ExistsNodeAtPos(MAP, node->x, node->y-1) && ExistsNodeAtPos(MAP, node->x+1, node->y-1)) {
        add_neighbor(neighbors, GetIdAtPos(MAP, node->x+1, node->y-1), sqrt(2));
    }
    if (ExistsNodeAtPos(MAP, node->x+1, node->y) && ExistsNodeAtPos(MAP, node->x, node->y+1) && ExistsNodeAtPos(MAP, node->x+1, node->y+1)) {
        add_neighbor(neighbors, GetIdAtPos(MAP, node->x+1, node->y+1), sqrt(2));
    }
}

void PrintPath(path *path) {
    printf("Path found!\n");
    printf("Number of nodes = %zu\n", path->count);
    printf("Total cost = %f\n", path->cost);
    for (int i = 0; i < path->count; i++) {
        printf("[%d,%d]\n", GetNodeById(MAP, path->nodeIds[i])->x, GetNodeById(MAP, path->nodeIds[i])->y);
    }
    printf("\n");
}

typedef struct {
    int id;
    char filename[256];
    int width, height;
    int start_x, start_y;
    int target_x, target_y;
    double cost;
} MapScen;

void printMapScen(MapScen entry) {
        printf("Mapa: %s\n", entry.filename);
        printf("ID: %d\n", entry.id);
        printf("Archivo: %s\n", entry.filename);
        printf("Dimensiones: %dx%d\n", entry.width, entry.height);
        printf("Inicio: (%d, %d)\n", entry.start_x, entry.start_y);
        printf("Meta: (%d, %d)\n", entry.target_x, entry.target_y);
        printf("Coste: %.8f\n", entry.cost);
}

static volatile int keepRunning = 1;

void intHandler(int dummy) {
    keepRunning = 0;
}


int main(int argc, char const *argv[])
{
    if (argc != 2) {
        perror("Uso: <exe> map.scen");
        return 1;
    }

    FILE *file = fopen(argv[1], "r");
    if (!file) {
        perror("No se pudo abrir el fichero del escenario");
        return 1;
    }

    clock_t start, end;
    double cpu_time_used;
    start = clock();

    int total_succeed = 0;
    int total_failed = 0;
    char map_file[256] = "";
    MapScen entry;
    fscanf(file, "%*[^\n]\n");
    signal(SIGINT, intHandler);
    while (keepRunning && fscanf(file, "%d\t%255s\t%d\t%d\t%d\t%d\t%d\t%d\t%lf\n",
                  &entry.id, entry.filename, &entry.width, &entry.height,
                  &entry.start_x, &entry.start_y, &entry.target_x, &entry.target_y,
                  &entry.cost) == 9)
    {
        if (strcmp(map_file, entry.filename) != 0) {
            strcpy(map_file, entry.filename);
            char map_dir[256] = "../maze-map/";
            if (MAP) FreeMap(MAP);
            MAP = LoadMap(strcat(map_dir, map_file));
            if (!MAP) {
                perror("Error al cargar el fichero del mapa");
                return 1;
            }
        }

        if (!ExistsNodeAtPos(MAP, entry.start_x, entry.start_y)) {
            perror("El nodo de inicio no es válido");
            return 1;
        } else if (!ExistsNodeAtPos(MAP, entry.target_x, entry.target_y)) {
            perror("El nodo objetivo no es válido");
            return 1;
        }

        AStarSource source = {MAP->count, &GetNeighbors, &DiagonalHeuristic};
        id_t start = GetIdAtPos(MAP, entry.start_x, entry.start_y);
        id_t target = GetIdAtPos(MAP, entry.target_x, entry.target_y);

        path *path = find_path_omp(&source, start, target, 4);

        printf("%d-", entry.id);
        if (fabs(path->cost - entry.cost) < 1e-4) {
            printf("[OK] Coste esperado: %.8f, Coste encontrado: %.8f\n", entry.cost, path->cost);
            total_succeed++;
        }
        else {
            printf("[Error] Coste esperado: %.8f, Coste encontrado: %.8f\n", entry.cost, path->cost);
            total_failed++;
        }

        path_destroy(path);
    }

    if (MAP) FreeMap(MAP);

    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("\nResultados:\n");
    printf("Tiempo total: %.2f segundos\n", cpu_time_used);
    printf("Total de mapas: %d\n", total_succeed + total_failed);
    printf("Total de exitos: %d\n", total_succeed);
    printf("Total de fallos: %d\n", total_failed);
    return 0;
}
