#include <stdlib.h>
#include "AStar.h"

typedef struct __Node *Node;
typedef struct __Neighbor *Neighbor;
typedef struct __NeighborsList *NeighborsList;
typedef struct __ListItem *ListItem;
typedef struct __ListItem *ClosedSet;
typedef struct __CostList *CostList;
typedef struct __OpenSet *OpenSet;

struct __Node {
    float h_value;
    float g_value;
    unsigned isOpen:1;
    unsigned isClosed:1;
    unsigned isGoal:1;
    Node parent;
    NeighborsList neighbors;
};

struct __NeighborsList {
    int count;
    Neighbor first;
};

struct __Neighbor {
    float cost;
    Node node;
    Neighbor next;
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

Node CreateNode() {
    Node n = malloc(sizeof(struct __Node));
    n->neighbors = malloc(sizeof(struct __NeighborsList));
    n->neighbors->count = 0;
    return n;
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

void AddNeighbor(Node node, Node neighbor, float cost) {
    Neighbor newNeighbor = CreateNeighbor(neighbor, cost);
    newNeighbor->next = node->neighbors->first;
    node->neighbors->first = newNeighbor;
    node->neighbors->count++;
    return;
}

NeighborsList GetNeighbors(Node n) {
    return n->neighbors;
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
    OpenSet open = maloc(sizeof(struct __OpenSet));
    open->count = 0;
    open->set = NULL;
}

void AddNodeToOpenSet(OpenSet open, Node n) {
    if (n->isOpen) return;
    n->isOpen = 1;
    float fcost = n->h_value + n->g_value;
    ListItem ni = CreateListItem(n);

    if (!open->set) {
        CostList nl = CreateCostList(fcost);
        nl->nodes = ni;
        nl->next = open;
        open->set = nl;
        open->count++;
        return;
    }

    CostList currentList = open->set;
    CostList previousList = NULL;
    while (currentList && currentList->cost < fcost) {
        previousList = currentList;
        currentList = currentList->next;
    }

    if (!currentList || currentList->cost > fcost) {
        CostList nl = CreateCostList(fcost);
        nl->nodes = ni;
        nl->next = currentList;
        previousList->next = nl;
        open->count++;
        return;
    }

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

void AddNodeToClosedSet(ClosedSet closed, Node n) {
    if (n->isClosed) return;
    n->isClosed = 1;

    ListItem ni = CreateListItem(n);
    ni->next = closed;
    closed = ni;
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

Node FindPath(AStarSource source, Node start, Node goal) {
    OpenSet open = CreateOpenSet();
    ClosedSet closed = NULL;

    AddNodeToOpenSet(open, start);

    while(open->count > 0) {
        Node currentNode = GetFirstFromOpenSet(open);
        if (currentNode = goal) return currentNode;
        for (int i = 0; i < currentNode->neighbors->count; i++) {
            Neighbor nb = currentNode->neighbors->first + i * sizeof(Neighbor);
            if (nb->node->isClosed) continue;
            float newCostToNeighbor = currentNode->g_value + source.Heuristic(currentNode, nb->node);
            if (newCostToNeighbor < nb->node->g_value || !nb->node->isOpen) {
                nb->node->g_value = newCostToNeighbor;
                nb->node->h_value = source.Heuristic(nb->node, goal);
                nb->node->parent = currentNode;
                if (!nb->node->isOpen) AddNodeToOpenSet(open, nb->node);
            }
        }
    }
}