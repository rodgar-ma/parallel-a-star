#include <stdlib.h>
#include "AStar.h"

typedef struct __Node *Node;
typedef struct __Neighbor *Neighbor;
typedef struct __VisitedNodes *VisitedNodes;
typedef struct __ListItem *ListItem;
typedef struct __CostList *CostList;
typedef struct __OpenSet *OpenSet;

struct __Node {
    void *node;
    float h_value;
    float g_value;
    unsigned isOpen:1;
    unsigned isClosed:1;
    unsigned isGoal:1;
    Node parent;
};

struct __Neighbor {
    float cost;
    Node node;
    Neighbor next;
};

struct __NeighborList {
    int count;
    Neighbor first;
};

struct __ListItem{
    Node node;
    ListItem next;
};

struct __CostList {
    float cost;
    ListItem nodes;
    CostList next;
};

struct __OpenSet {
    int count;
    CostList set;
};

Node CreateNode(void *node) {
    Node n = malloc(sizeof(struct __Node));
    n->node = node;
    n->parent = NULL;
    n->isOpen = 0;
    n->isClosed = 0;
    n->isGoal = 0;
    return n;
}

Node GetNode(VisitedNodes visited, void *node) {
    return NULL;
}

Neighbor CreateNeighbor(Node n, float cost) {
    Neighbor nb = malloc(sizeof(struct __Neighbor));
    nb->node = n;
    nb->cost = cost;
    return nb;
}

void SetNodeGoal(Node n) {
    n->isGoal = 1;
}

ListItem CreateListItem(Node n) {
    ListItem li = malloc(sizeof(struct __ListItem));
    li->node = n;
    return li;
}

CostList CreateCostList(float cost) {
    CostList cl = malloc(sizeof(struct __CostList));
    if (!cl) return NULL;
    cl->cost = cost;
    cl->nodes = NULL;
    cl->next = NULL;
    return cl;
}

OpenSet CreateOpenSet() {
    OpenSet open = malloc(sizeof(struct __OpenSet));
    open->count = 0;
    open->set = NULL;
}

void AddNodeToOpenSet(OpenSet open, Node n) {
    if (n->isOpen) return;
    n->isOpen = 1;
    float fcost = n->h_value + n->g_value;
    ListItem ni = CreateListItem(n);

    // Si el set está vacía creamos una nueva lista y lo insertamos.
    if (!open->set) {
        CostList nl = CreateCostList(fcost);
        nl->nodes = ni;
        nl->next = open;
        open->set = nl;
        open->count++;
        return;
    }

    // Buscamos la lista con coste fcost.
    CostList currentList = open->set;
    CostList previousList = NULL;
    while (currentList && currentList->cost < fcost) {
        previousList = currentList;
        currentList = currentList->next;
    }

    // Si no existe la creamos
    if (!currentList || currentList->cost > fcost) {
        CostList nl = CreateCostList(fcost);
        nl->nodes = ni;
        nl->next = currentList;
        previousList->next = nl;
        open->count++;
        return;
    }

    // Si existe buscamos el nodo
    ListItem currentItem = currentList->nodes;
    ListItem previousItem = NULL;
    while (currentItem && currentItem->node->h_value < n->h_value) {
        previousItem = currentItem;
        currentItem = currentItem->next;
    }
    
    if (!previousItem) {
        ni->next = currentList->nodes;
        currentList->nodes = ni;
    } else {
        ni->next = currentItem;
        previousItem->next = ni;
    }
    open->count++;
}

// A lo mejor no hace falta. Al tomar el primero se elimina.
void RemoveNodeFromOpenSet(OpenSet open, Node n) {
    if (!n->isOpen) return;
    n->isOpen = 0;

    CostList tmpList = open->set;
    CostList prevList = NULL;
    while (tmpList) {
        ListItem tmpNode = tmpList->nodes;
        ListItem prevNode = NULL;
        while (tmpNode) {
            if (tmpNode->node == n) {
                if (prevNode) {
                    prevNode->next = tmpNode->next;
                } else {
                    tmpList->nodes = tmpNode->next;
                }
                open->count--;
                return;
            }
            prevNode = tmpNode;
            tmpNode = tmpNode->next;
        }
        prevList = tmpList;
        tmpList = tmpList->next;
    }
}

Node GetFirstFromOpenSet(OpenSet open) {
    Node n = open->set->nodes->node;
    if (open->set->nodes->next) {
        open->set->nodes = open->set->nodes->next;
    } else {
        open->set = open->set->next;
    }
    return n;
}

Node FindPath(AStarSource * source, void *start, void *goal) {
    OpenSet open = CreateOpenSet();
    VisitedNodes visited = CreateVisitedNodes();

    Node startNode = GetNode(visited, start);
    Node goalNode = GetNode(visited, goal);

    startNode->h_value = source->Heuristic(startNode->node, goalNode->node);
    goalNode->isGoal = 1;

    AddNodeToOpenSet(open, startNode);

    while(open->count > 0) {

    }

}