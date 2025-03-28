#include "AStar.h"

typedef struct {
    int x;
    int y;
} Node;

int main(int argc, char const *argv[])
{
    void ** nodes = malloc(10 * sizeof(Node*));
    for (int i = 0; i < 10; i++) {
        nodes[i] = malloc(10 * sizeof(Node));
    }

    ASPathNodeSource source;
    ASNeighborList neighbors;

    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            ASNeighborListAdd(neighbors, nodes);
        }
        
    }
    

    return 0;
}
