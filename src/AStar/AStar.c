#include <stdlib.h>
#include <stdio.h>
#include "AStar.h"

typedef struct __Node *Node;
typedef struct __NodeRecord *NodeRecord;
typedef struct __NodeList *NodeList;
typedef struct __OpenSet *OpenSet;
typedef struct __ClosedSet *ClosedSet;

struct __Node {
    float h_value;
    float g_value;
    unsigned isOpen:1;
    unsigned isClosed:1;
    NodeRecord neighbors;
};

struct __NodeRecord {
    Node node;
    NodeRecord next;
};

struct __NodeList {
    float cost;
    NodeRecord nodes;
    NodeList next;
};

struct __OpenSet {
    NodeList set;
};

struct __ClosedSet {
    NodeRecord set;
};

static Node CreateNode(float hValue, float gValue)
{
    Node n = malloc(sizeof(struct __Node));
    if (!n) return NULL;
    n->h_value = hValue;
    n->g_value = gValue;
    n->isOpen = 0;
    n->isClosed = 0;
    n->neighbors = NULL;
    return n;
}

static NodeRecord CreateNodeRecord(Node n) {
    NodeRecord nr = malloc(sizeof(struct __NodeRecord));
    if (!nr) return NULL;
    nr->node = n;
    nr->next = NULL;
    return nr;
}

static NodeList CreateNodeList(float cost) {
    NodeList nl = malloc(sizeof(struct __NodeList));
    if (!nl) return NULL;
    nl->cost = cost;
    nl->nodes = NULL;
    nl->next = NULL;
    return nl;
}

static OpenSet CreateOpenSet() {
    OpenSet s = malloc(sizeof(struct __OpenSet));
    s->set = NULL;
    return s;
}

static ClosedSet CreateClosedSet() {
    ClosedSet s = malloc(sizeof(struct __ClosedSet));
    s->set = NULL;
    return s;
}

void AddNeighbor(Node n, Node new) {   
    NodeRecord nr = CreateNodeRecord(new);
    nr->next = n->neighbors;
    n->neighbors = nr;
}

/* No es necesario por el atributo isOpen */
int SetHasNode(OpenSet set, Node n) {
    NodeList list = set->set;
    if (!list || !n) return 0;
    NodeList actualList = list;
    while(actualList) {
        NodeRecord actualNode = actualList->nodes;
        while(actualNode) {
            if (actualNode->node == n) return 1;
            actualNode = actualNode->next;
        }
        actualList = actualList->next;
    }
    return 0;
}

void AddNodeToOpenSet(OpenSet set, Node n, float cost) {
    if (!n || n->isOpen) {
        return;
    } else {
        n->isOpen = 1;
    }

    NodeRecord nr = CreateNodeRecord(n);
    if (!set->set || set->set->cost > cost) {
        NodeList nl = CreateNodeList(cost);
        nl->nodes = nr;
        nl->next = set->set;
        set->set = nl;
        return;
    }
    
    NodeList prevList = set->set;
    while (prevList->next && prevList->next->cost < cost) {
        prevList = prevList->next;
    }
    
    if (!prevList->next || prevList->next->cost < cost) {
        NodeList nl = CreateNodeList(cost);
        nl->nodes = nr;
        nl->next = prevList->next;
        prevList->next = nl;
    } else {
        nr->next = prevList->next->nodes;
        prevList->next->nodes = nr;
    }
}

void RemoveNodeFromOpenSet(OpenSet set, Node n) {
    if (!n || !n->isOpen || !set->set) {
        return;
    } else {
        n->isOpen = 0;
    }

    NodeList tmpList = set->set;
    while (tmpList) {
        NodeRecord prev = tmpList->nodes;
        NodeRecord tmpNode = tmpList->nodes;
        while (tmpNode && tmpNode->node != n) {
            prev = tmpNode;
            tmpNode = tmpNode->next;
        }
        if (tmpNode) {
            prev->next = tmpNode->next;
            return;
        }
        tmpList = tmpList->next;
    }
}

void AddNodeToClosedSet(ClosedSet closed, Node n) {
    if (!n || n->isClosed) return;
    n->isClosed = 1;
    NodeRecord nr = CreateNodeRecord(n);
    nr->next = closed->set;
    closed->set = nr;
}

NodeRecord FindPath(Node source, Node goal) {
    OpenSet openSet = CreateOpenSet();
    ClosedSet closedSet = CreateClosedSet();
    return NULL;
}

void PrintSet(OpenSet set) {
    NodeList nl = set->set;
    printf("Node Set:\n");
    while(nl) {
        printf("\tCost %f\n", nl->cost);
        NodeRecord nr = nl->nodes;
        while(nr) {
            printf("\t\tNode, h=%f, g=%f\n", nr->node->h_value, nr->node->g_value);
            nr = nr->next;
        }
        printf("\n");
        nl = nl->next;
    }
}


/*******************************************************************/

int main(int argc, char const *argv[])
{
    Node n1 = CreateNode(1, 1);
    Node n2 = CreateNode(2, 2);
    Node n3 = CreateNode(3, 3);
    Node n4 = CreateNode(4, 4);
    Node n5 = CreateNode(5, 5);
    Node n6 = CreateNode(6, 6);
    
    AddNeighbor(n1, n2);
    AddNeighbor(n1, n5);
    AddNeighbor(n1, n3);
    AddNeighbor(n1, n4);
    AddNeighbor(n1, n6);

    OpenSet open = CreateOpenSet();
    AddNodeToOpenSet(open, n1, n1->g_value + n1->h_value);
    AddNodeToOpenSet(open, n2, n2->g_value + n2->h_value);
    AddNodeToOpenSet(open, n3, n3->g_value + n3->h_value);
    AddNodeToOpenSet(open, n4, n4->g_value + n4->h_value);
    AddNodeToOpenSet(open, n5, n5->g_value + n5->h_value);
    AddNodeToOpenSet(open, n6, n6->g_value + n6->h_value);

    PrintSet(open);
    
    return 0;
}

