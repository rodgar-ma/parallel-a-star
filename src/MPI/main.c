#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <signal.h>
#include <time.h>
#include "MapUtils.h"
#include "astar.h"

typedef struct {
    int id;
    char filename[256];
    int width, height;
    int start_x, start_y;
    int target_x, target_y;
    float cost;
} MapScen;

static volatile int keepRunning = 1;

void intHandler(int dummy) {
    keepRunning = 0;
}

Map MAP;

char MAP_SCEN_FILENAME[256];

int main(int argc, char const *argv[])
{
    // if (argc != 3) {
    //     perror("Uso: <exe> num_threads map.scen");
    //     return 1;
    // }

    // int num_threads = atoi(argv[1]);

    FILE *file = fopen(argv[1], "r");
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

    double cpu_time_used;
    clock_t start = clock();
    while (keepRunning && fscanf(file, "%d\t%255s\t%d\t%d\t%d\t%d\t%d\t%d\t%f\n",
                  &entry.id, entry.filename, &entry.width, &entry.height,
                  &entry.start_x, &entry.start_y, &entry.target_x, &entry.target_y,
                  &entry.cost) == 9)
    {
        if (strcmp(map_file, entry.filename) != 0) {
            strcpy(map_file, entry.filename);
            strcat(MAP_SCEN_FILENAME, entry.filename);
            char maps_dir[256] = "../maps/";
            if (MAP) FreeMap(MAP);
            MAP = LoadMap(strcat(maps_dir, map_file));
            if (!MAP) {
                perror("Error al cargar el fichero del mapa");
                return 1;
            }
        }

        if (!ExistsNodeAtPos(MAP, entry.start_x, entry.start_y)) {
            printf("%d-[Error] Nodo de inicio inválido (%d, %d)\n", entry.id, entry.start_x, entry.start_y);
            total_failed++;
            continue;
        }
        if (!ExistsNodeAtPos(MAP, entry.target_x, entry.target_y)) {
            printf("%d-[Error] Nodo objetivo inválido (%d, %d)\n", entry.id, entry.target_x, entry.target_y);
            total_failed++;
            continue;
        }
        
        
        AStarSource source = {MAP->width*MAP->height, &GetNeighbors, &DiagonalHeuristic};
        int s_id = GetIdAtPos(MAP, entry.start_x, entry.start_y);
        int t_id = GetIdAtPos(MAP, entry.target_x, entry.target_y);
        
        printf("Starting search...\n");
        path *path = astar_search(&source, s_id, t_id, &cpu_time_used);

        if (!path) {
            printf("[Error] No se encontró ningún camino de (%d, %d) a (%d, %d)\n",
                   entry.start_x, entry.start_y, entry.target_x, entry.target_y);
            total_failed++;
            continue;
        }

        printf("%d-", entry.id);
        if (fabs(path->cost - entry.cost) < 1) {
            printf("[OK] ");
            total_succeed++;
        } else {
            printf("[Error] ");
            total_failed++;
        }
        printf("Coste esperado: %.8f, Coste encontrado: %.8f, Tiempo: %lf ms\n", entry.cost, path->cost, 10e3 * cpu_time_used);

        path_destroy(path);
    }

    clock_t end = clock();

    if (MAP) FreeMap(MAP);

    printf("\nResultados:\n");
    printf("Tiempo total: %ld segundos\n", (end - start) / CLOCKS_PER_SEC);
    printf("Total de mapas: %d\n", total_succeed + total_failed);
    printf("Total de exitos: %d\n", total_succeed);
    printf("Total de fallos: %d\n", total_failed);
    return 0;
}
