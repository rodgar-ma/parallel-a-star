#include <stdlib.h>
#include "AStar.h"

typedef struct __Node *Node;
typedef struct __Neighbor *Neighbor;
typedef struct __ListItem *ListItem;
typedef struct __CostList *CostList;
typedef struct __CostList *OpenSet;
typedef struct __ListItem *ClosedSet;

struct __Node {
    void *node;
    void *parent;
    float h_value;
    float g_value;
    unsigned isOpen:1;
    unsigned isClosed:1;
    unsigned isGoal:1;
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

Node CreateNode(void *node) {
    Node n = malloc(sizeof(struct __Node));
    n->node = node;
    n->parent = NULL;
    n->isOpen = 0;
    n->isClosed = 0;
    n->isGoal = 0;
    n->neighbors = malloc(sizeof(struct __NeighborsList));
    n->neighbors->count = 0;
    return n;
}

ListItem CreateListItem(Node n) {
    ListItem ni = malloc(sizeof(ListItem));
    ni->node = n;
    return ni;
}

void AddNeighbor(NeighborsList neighbors, void *node, float cost) {
    Node n = GetNode(node);
    NeighborsList newNeighbor = malloc(sizeof(NeighborsList));
    newNeighbor->cost = cost;
    newNeighbor->node = n;
    newNeighbor->next = neighbors;
    neighbors = newNeighbor;
}

CostList CreateCostList(float cost) {
    CostList nl = malloc(sizeof(CostList));
    if (!nl) return NULL;
    nl->cost = cost;
    nl->nodes = NULL;
    nl->next = NULL;
    return nl;
}

void AddNodeToOpenSet(CostList open, Node n) {
    if (n->isOpen) return;
    n->isOpen = 1;
    float fcost = n->h_value + n->g_value;
    ListItem ni = CreateListItem(n);

    if (!open) {
        CostList nl = CreateCostList(fcost);
        nl->nodes = ni;
        nl->next = open;
        open = nl;
        return;
    }

    CostList currentList = open;
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
}

void RemoveNodeFromOpenSet(CostList set, Node n) {
    if (!n->isOpen) return;
    n->isOpen = 0;

    CostList tmpList = set;
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
    Node n = open->nodes->node;

    if (open->nodes->next) {
        open->nodes = open->nodes->next;
    } else {
        open = open->next;
    }

    return n;
}

Path FindPath(const AStarSource * source, void *start, void *goal) {
    OpenSet open = NULL;
    ClosedSet closed = NULL;


}