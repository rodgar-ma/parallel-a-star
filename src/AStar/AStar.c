#include <stdlib.h>
#include "AStar.h"

typedef struct __NodeRecord *NodeRecord;
typedef struct __NeighborRecord *NeighborRecord;
typedef struct __ListItem *ListItem;
typedef struct __CostList *CostList;
typedef struct __CostList *OpenSet;
typedef struct __ListItem *ClosedSet;

struct __NodeRecord {
    void *node;
    void *parent;
    float h_value;
    float g_value;
    unsigned isOpen:1;
    unsigned isClosed:1;
    unsigned isGoal:1;
    NeighborRecord neighbors;
};

struct __NeighborRecord {
    float cost;
    NeighborRecord next;
};

struct __ListItem{
    NodeRecord node;
    ListItem next;
};

struct __CostList {
    float cost;
    ListItem nodes;
    CostList next;
};


NodeRecord CreateNodeRecord(void *node) {
    NodeRecord nr = malloc(sizeof(NodeRecord));
    if (!nr) return NULL;
    nr->node = node;
    nr->parent = NULL;
    nr->isOpen = 0;
    nr->isClosed = 0;
    nr->isGoal = 0;
    return nr;
}

ListItem CreateListItem(NodeRecord nr) {
    ListItem ni = malloc(sizeof(ListItem));
    ni->node = nr;
    return ni;
}

void AddNeighbor(void * node, void * neighbor, float cost) {
    
}

CostList CreateCostList(float cost) {
    CostList nl = malloc(sizeof(CostList));
    if (!nl) return NULL;
    nl->cost = cost;
    nl->nodes = NULL;
    nl->next = NULL;
    return nl;
}

void AddNodeRecordToOpenSet(CostList open, NodeRecord nr) {
    if (nr->isOpen) return;
    nr->isOpen = 1;
    float fcost = nr->h_value + nr->g_value;
    ListItem ni = CreateListItem(nr);

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
    while (currentItem && currentItem->node->h_value < nr->h_value) {
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

void RemoveNodeFromOpenSet(CostList set, NodeRecord nr) {
    if (!nr->isOpen) return;
    nr->isOpen = 0;

    CostList tmpList = set;
    CostList prevList = NULL;
    while (tmpList) {
        ListItem tmpNode = tmpList->nodes;
        ListItem prevNode = NULL;
        while (tmpNode) {
            if (tmpNode->node == nr) {
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

void AddNodeToClosedSet(ClosedSet closed, NodeRecord nr) {
    if (nr->isClosed) return;
    nr->isClosed = 1;

    ListItem ni = CreateListItem(nr);
    ni->next = closed;
    closed = ni;
}

NodeRecord GetFirst(OpenSet open) {
    NodeRecord nr = open->nodes->node;

    if (open->nodes->next) {
        open->nodes = open->nodes->next;
    } else {
        open = open->next;
    }

    return nr;
}

Path FindPath(const AStarSource * source, void *start, void *goal) {
    OpenSet open = NULL;
    ClosedSet closed = NULL;


}