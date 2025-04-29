#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <math.h>
#include <time.h>
#include "astar.h"
#include "MapReader.h"
#include "utils.h"

typedef struct {
    int id;
    char filename[256];
    int width, height;
    int start_x, start_y;
    int target_x, target_y;
    double cost;
} MapScen;

static volatile int keepRunning = 1;

void intHandler(int dummy) {
    keepRunning = 0;
}

int main(int argc, char const *argv[])
{
    if (argc != 3) {
        perror("Uso: <exe> num_threads map.scen");
        return 1;
    }

    int num_threads = atoi(argv[1]);
    FILE *file = fopen(argv[2], "r");
    if (!file) {
        perror("No se pudo abrir el fichero del escenario");
        return 1;
    }

    signal(SIGINT, intHandler);
    
    char map_file[256] = "";
    MapScen entry;
    fscanf(file, "%*[^\n]\n");

    int total_succeed = 0;
    int total_failed = 0;
    clock_t start, end;
    double cpu_time_used;
    start = clock();
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
        astar_id_t start = GetIdAtPos(MAP, entry.start_x, entry.start_y);
        astar_id_t target = GetIdAtPos(MAP, entry.target_x, entry.target_y);

        path *path = find_path_omp(&source, start, target, num_threads);

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
