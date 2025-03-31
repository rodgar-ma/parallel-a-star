#include <stdlib.h>
#include <stdio.h>
#include "AStar.h"
#include "MapReader.h"

// // Conectar nodos vecinos
    // for (int y = 0; y < height; y++) {
    //     for (int x = 0; x < width; x++) {
    //         if (map->grid[y][x]) {
    //             if (x > 0 && map->grid[y][x - 1]) AddNeighbor(map->grid[y][x], map->grid[y][x - 1]);
    //             if (x < width - 1 && map->grid[y][x + 1]) AddNeighbor(map->grid[y][x], map->grid[y][x + 1]);
    //             if (y > 0 && map->grid[y - 1][x]) AddNeighbor(map->grid[y][x], map->grid[y - 1][x]);
    //             if (y < height - 1 && map->grid[y + 1][x]) AddNeighbor(map->grid[y][x], map->grid[y + 1][x]);
    //         }
    //     }
    // }

int main(int argc, char const *argv[])
{
    printf("\n");
    char* filename = "./maze-map/maze512-1-0.map";
    Map *map = LoadMap(filename);
    
    void *source = map->grid[1][1];
    void *goal = map->grid[511][511];

    FreeMap(map);

    return 0;
}
