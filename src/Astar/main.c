#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <signal.h>
#include <time.h>
#include "MapUtils.h"
#include "astar.h"
#include "spastar.h"
#include "hdastar.h"

typedef struct {
    int id;
    char map_file[256];
    int width, height;
    int start_x, start_y;
    int target_x, target_y;
    float cost;
} MapScen;

static volatile int keepRunning = 1;

void intHandler(int dummy) {
    keepRunning = 0;
}

void help() {
    printf("Uso:\n");
    printf("A*: main seq file.scen\n");
    printf("SPA*: main spa n_threads file.scen\n");
    printf("HDA*: main hda n_threads file.scen\n\n");
}

Map *MAP;

int main(int argc, char *argv[])
{

    signal(SIGINT, intHandler);

    int type;
    char *filename;
    int n_threads;

    if (strcmp(argv[1], "seq") == 0) {
        type = 0;
        filename = argv[2];
        n_threads = 1;
    } else if (strcmp(argv[1], "spa") == 0) {
        type = 1;
        n_threads = atoi(argv[2]);
        filename = argv[3];
    } else if (strcmp(argv[1], "hda") == 0) {
        type = 2;
        n_threads = atoi(argv[2]);
        filename = argv[3];
    }
    
    FILE *scen_file = fopen(filename, "r");
    if (scen_file == NULL) {
        printf("Error: file %s not found\n", filename);
        help();
        return EXIT_FAILURE;
    }
    
    fscanf(scen_file, "%*[^\n]\n");

    MapScen entry;
    char map_file[256];
    
    int total_succeed = 0;
    int total_failed = 0;
    int map_succeed = 0;
    int map_failed = 0;

    double cpu_time_used;
    double acumulated_time = 0;
    clock_t start = clock();
    int last_id = -1;

    while (keepRunning && fscanf(scen_file, "%d\t%255s\t%d\t%d\t%d\t%d\t%d\t%d\t%f\n",
                  &entry.id, entry.map_file, &entry.width, &entry.height,
                  &entry.start_x, &entry.start_y, &entry.target_x, &entry.target_y,
                  &entry.cost) == 9)
    {
        // If not same map then load the new map.
        if (strcmp(map_file, entry.map_file) != 0 || entry.id != last_id) {

            if (map_succeed + map_failed != 0) {
                printf("Mapa %s terminado.\n", map_file);
                printf("Tiempo total: %lf ms\n", acumulated_time);
                printf("Tiempo medio por mapa: %lf\n\n", acumulated_time / (map_succeed + map_failed));
            }

            map_succeed = 0;
            map_failed = 0;
            acumulated_time = 0;

            strcpy(map_file, entry.map_file);
            last_id = entry.id;
            char maps_dir[256] = "../maps/";
            if (MAP) FreeMap(MAP);
            MAP = LoadMap(strcat(maps_dir, map_file));
            if (MAP == NULL) {
                printf("Error al cargar el fichero del mapa: %s\n", map_file);
                return EXIT_FAILURE;
            }
        }

        // Start and Goal should exists.
        int error = 0;
        if (!is_walkable(entry.start_x, entry.start_y)) {
            printf("%d-[Error] Nodo de inicio inválido (%d, %d)\n", entry.id, entry.start_x, entry.start_y);
            error = 1;
        }
        if (!is_walkable(entry.target_x, entry.target_y)) {
            printf("%d-[Error] Nodo objetivo inválido (%d, %d)\n", entry.id, entry.target_x, entry.target_y);
            error = 1;
        }
        if (error) {
            map_failed++;
            total_failed++;
            continue;
        }
        
        
        AStarSource source = {MAP->width*MAP->height, &GetNeighbors, &DiagonalHeuristic};
        int s_id = xyToID(entry.start_x, entry.start_y);
        int t_id = xyToID(entry.target_x, entry.target_y);
        
        path *path = NULL;

        if (type == 0) {
            path = astar_search(&source, s_id, t_id, &cpu_time_used);
        } else if (type == 1) {
            path = spastar_search(&source, s_id, t_id, n_threads, &cpu_time_used);
        }  else if (type == 2) {
            path = hdastar_search(&source, s_id, t_id, n_threads, &cpu_time_used);
        }
        

        if (!path) {
            // printf("[Error] No se encontró ningún camino de (%d, %d) a (%d, %d)\n", entry.start_x, entry.start_y, entry.target_x, entry.target_y);
            map_failed++;
            total_failed++;
            continue;
        }

        // printf("%d-", entry.id);
        if (fabs(path->cost - entry.cost) < 1) {
            // printf("[OK] ");
            map_succeed++;
            total_succeed++;
        } else {
            // printf("[Error] ");
            map_failed++;
            total_failed++;
        }
        // printf("Coste esperado: %.8f, Coste encontrado: %.8f, Tiempo: %lf ms\n", entry.cost, path->cost, 10e3 * cpu_time_used);
        acumulated_time += 10e3 * cpu_time_used;
        path_destroy(path);
    }

    clock_t end = clock();

    if (MAP) FreeMap(MAP);

    if (map_succeed + map_failed != 0) {
        printf("Mapa %s terminado.\n", map_file);
        printf("Tiempo total: %lf ms\n", acumulated_time);
        printf("Tiempo medio por mapa: %lf\n\n", acumulated_time / (map_succeed + map_failed));
    }

    printf("\nResultados:\n");
    printf("Tiempo total: %ld segundos\n", (end - start) / CLOCKS_PER_SEC);
    printf("Total de mapas: %d\n", total_succeed + total_failed);
    printf("Total de exitos: %d\n", total_succeed);
    printf("Total de fallos: %d\n", total_failed);
    printf("Tiempo medio por mapa: %f\n\n", acumulated_time / (total_succeed + total_failed));

    return EXIT_SUCCESS;
}
