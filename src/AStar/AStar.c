#include <stdlib.h>
#include <stdio.h>
#include "AStar.h"

typedef struct __Node *Node;
typedef struct __NodeRecord *NodeRecord;
typedef struct __NodeList *NodeList;

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

int ListHasNode(NodeList list, Node n) {
    if (!list || !n) return 0;
    NodeList actualList = list;
    while(actualList) {
        NodeRecord actualNode = actualList->nodes;
        while(actualNode) {
            if (actualNode->node == n) return 1;
            actualNode = actualNode->next;
        }
    }
    return 0;
}

void AddNodeToList(NodeList list, Node n, float cost) {
    NodeRecord nr = CreateNodeRecord(n);
    if (!list || list->cost > cost) {
        NodeList nl = CreateNodeList(cost);
        nl->nodes = nr;
        nl->next = list;
        list = nl;
        return;
    }
    
    NodeList prevList = list;
    while (prevList->next && prevList->next->cost < cost) {
        prevList = prevList->next;
    }
    
    if (!prevList->next) prevList->next = CreateNodeList(cost);

    nr->next = prevList->next->nodes;
    prevList->next->nodes = nr;
}

void RemoveNodeFromList(NodeList list, Node n) {
    if (!list || !n) return;
    NodeList tmpList = list;
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

void AddNodeToOpenSet(NodeList open, Node n) {
    if (ListHasNode(open, n)) return;
    AddNodeToList(open, n, n->h_value + n->g_value);
    n->isOpen = 1;
}

void AddNeighbor(Node n, Node new) {   
    NodeRecord nr = CreateNodeRecord(new);
    nr->next = n->neighbors;
    n->neighbors = nr;
}

NodeRecord FindPath(Node source, Node goal) {
    NodeList openSet = NULL;
    NodeList closedSet = NULL;
    return NULL;
}

void PrintList(NodeList list) {
    NodeList nl = list;
    printf("Node List:\n");
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

    NodeList openset = NULL;
    AddNodeToOpenSet(openset, n1);
    AddNodeToOpenSet(openset, n2);
    AddNodeToOpenSet(openset, n3);
    AddNodeToOpenSet(openset, n4);
    AddNodeToOpenSet(openset, n5);
    AddNodeToOpenSet(openset, n6);

    PrintList(openset);
    
    return 0;
}

