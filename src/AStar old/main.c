#include <stdlib.h>
#include <stdio.h>
#include "AStar.h"
#include "MapReader.h"

    

int main(int argc, char const *argv[])
{
    printf("\n");
    char* filename = "./maze-map/maze512-1-0.map";
    Map map = LoadMap(filename);

    AStarSource source;
    

    FreeMap(map);
    return 0;
}
