#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct __Node *Node;
typedef struct __Neighbor *Neighbor;
typedef struct __NeighborsList *NeighborsList;
typedef struct __ListItem *ListItem;
typedef struct __CostList *CostList;
typedef struct __OpenSet *OpenSet;
typedef struct __ListItem *ClosedSet;
typedef struct __VisitedNodes *VisitedNodes;
typedef struct __Map *Map;

struct __Node {
    int x;
    int y;
    Node parent;
    float h_value;
    float g_value;
    unsigned isOpen:1;
    unsigned isClosed:1;
    unsigned isGoal:1;
};

struct __Neighbor {
    float cost;
    Node node;
    Neighbor next;
};

struct __NeighborsList {
    int count;
    Neighbor list;
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

struct __Map {
    int width;
    int height;
    Node **grid;
};

Map CreateMap(const char *filename) {
    FILE *file;
    int err;

    if ((err = fopen(&file, filename)) != 0) {
        perror("Error abriendo el archivo");
        return NULL;
    }

    int width = 0, height = 0;
    char buffer[1024];

    // Leer dimensiones
    while (fgets(buffer, sizeof(buffer), file)) {
        if (strncmp(buffer, "width", 5) == 0) {
            if ((err = sscanf(buffer, "width %d", &width)) != 1) {
                perror("Error abriendo el archivo");
                return NULL;
            }
        } else if (strncmp(buffer, "height", 6) == 0) {
            if ((err = sscanf(buffer, "height %d", &height)) != 1) {
                perror("Error abriendo el archivo");
                return NULL;
            }
        } else if (strncmp(buffer, "map", 3) == 0) {
            break;  // Inicia la lectura del mapa
        }
    }

    if (width == 0 || height == 0) {
        printf("Error: dimensiones no especificadas correctamente.\n");
        fclose(file);
        return NULL;
    }

    // Reservar memoria para el mapa
    Map map = malloc(sizeof(struct __Map));
    map->width = width;
    map->height = height;
    map->grid = calloc(height, sizeof(Node*));

    // Leer el mapa
    for (int y = 0; y < height; y++) {
        fgets(buffer, sizeof(buffer), file);
        map->grid[y] = calloc(width, sizeof(Node));

        for (int x = 0; x < width; x++) {
            if (buffer[x] == '.') {
                map->grid[y][x] = CreateNode();
            } else {
                map->grid[y][x] = NULL;
            }
        }
    }

    fclose(file);

    return map;
}

void FreeMap(Map map) {
    for (int y = 0; y < map->height; y++) {
        for (int x = 0; x < map->width; x++) {
            if (map->grid[y][x]) free(map->grid[y][x]);
        }
        free(map->grid[y]);
    }
    free(map->grid);
    free(map);
}

Node CreateNode() {
    Node n = malloc(sizeof(struct __Node));
    if (!n) return NULL;
    n->parent = NULL;
    n->isOpen = 0;
    n->isClosed = 0;
    n->isGoal = 0;
    return n;
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

void AddNodeToOpenSet(OpenSet open, Node nr) {
    if (nr->isOpen) return;
    nr->isOpen = 1;
    float fcost = nr->h_value + nr->g_value;
    ListItem ni = CreateListItem(nr);

    if (!open->set) {
        CostList nl = CreateCostList(fcost);
        nl->nodes = ni;
        nl->next = open->set;
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
    open->count++;
}

void RemoveNodeFromOpenSet(OpenSet open, Node nr) {
    if (!nr->isOpen) return;
    nr->isOpen = 0;
    
    CostList tmpList = open->set;
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

Node GetFirstFromOpenSet(OpenSet open) {
    Node nr = open->set->nodes->node;
    CostList set = open->set;
    if (set->nodes->next) {
        set->nodes = set->nodes->next;
    } else {
        set = set->next;
    }
    return nr;
}

void AddNodeToClosedSet(ClosedSet closed, Node nr) {
    if (nr->isClosed) return;
    nr->isClosed = 1;

    ListItem ni = CreateListItem(nr);
    ni->next = closed;
    closed = ni;
}

void SetHeuristics(Map map, Node start, Node goal) {
    for (int y = 0; y < map->height; y++) {
        for (int x = 0; x < map->width; x++) {
            map->grid[y][x]->h_value = Heuristic(start, map->grid[y][x]);
            map->grid[y][x]->g_value = Heuristic(map->grid[y][x], goal);
        }
    }
}

NeighborsList GetNeighbors(Map map, Node node) {
    NeighborsList neighbors = malloc(sizeof(struct __NeighborsList));
    neighbors->count = 0;
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            if (map->grid[node->y + i][node->x + j]) {
                
            }
        }
    }
    return neighbors;
}

Node FindPath(Map map, Node start, Node goal) {
    OpenSet open = CreateOpenSet();
    ClosedSet closed = NULL;

    SetHauristics(map, start, goal);

    AddNodeToOpenSet(open, start);

    while(open->count > 0) {
        Node current = GetFirstFromOpenSet(open);
        AddNodeToClosedSet(closed, current);
        if (current == goal) return current;

        NeighborsList neighbors = GetNeighbors(current);
        for (int i = 0; i < neighbors->count; i++) {

        }
        
    }

}

int main(int argc, char const *argv[])
{
    Map map = CreateMap("../AStar/maze-map/maze512-1-0.map");

    return 0;
}
