#include <stdlib.h>
#include "AStar.h"

typedef struct {
<<<<<<< Updated upstream
    
} NodeRecord;
=======
    size_t capacity;
    size_t count;
    int *costs;
    void **nodes;
} NeighborsList;

typedef struct {
    void *node;
    Node *parent;
    int gCost;
    int fCost;
    unsigned isOpen:1;
    unsigned isClosed:1;
} Node;

typedef struct HashTable *VisitedNodes;
typedef struct PriorityList *OpenList;

static inline NeighborsList* CreateNeighborsList() {
    NeighborsList *list = malloc(sizeof(NeighborsList));
    list->capacity = 0;
    list->costs = 0;
    return list;
}

void AddNeighbor(NeighborsList *list, void *node, int cost) {
    if (list->count == list->capacity) {
        list->capacity = 1 + (2 * list->capacity);
        list->costs = realloc(list->costs, list->capacity * sizeof(int));
        list->nodes = realloc(list->nodes, list->capacity * sizeof(void *));
    }
    list->costs[list->count] = cost;
    list->nodes[list->count] = node;
    list->count++;
}

Node GetNode(VisitedNodes *visited, void *node) {

}

Node GetFirstFromOpen(OpenList *open) {

}

void FindPath(AStarSource *source, void *start, void *goal) {
    OpenList *open = CreateNeighborsList();
    VisitedNodes *visited = CreateVisitedNodes();
    NeighborsList *neighborsList = CreateNeighborsList();

    Node current = GetNode(visited, start);
    current.gCost = 0;
    current.fCost = source->Heuristic(start, goal);

    AddNodeToOpenSet(open, current);

    while(HasOpenNodes(open)) {
        Node current = GetFirstFromOpen(open);
        if (current.node = goal) {
            return;
        }
        SetNodeClosed(visited, current);
        source->GetNeighbors(neighborsList, current.node);
        for (size_t i = 0; i < neighborsList->count; i++) {
            Node neighbor = GetNode(visited, neighborsList->nodes[i]);
            int newCost = neighborsList->costs[i] + source->Heuristic(neighbor.node, goal);
            if (neighbor.isClosed && newCost < neighbor.fCost) {
                neighbor.fCost = newCost;
                neighbor.isClosed = 0;
            } else {
                neighbor.gCost = current.gCost + neighborsList->costs[i];
                neighbor.fCost = newCost;
            }
            AddNodeToOpenSet(open, neighbor);
        }
    }
}
>>>>>>> Stashed changes
