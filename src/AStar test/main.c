#include <stdlib.h>
#include "AStar.h"
#include "MapReader.h"

int main(int argc, char const *argv[])
{
    const char* filename = "./maze-map/maze512-1-0.map";
    Map map = LoadMap(filename);
    
    return 0;
}
