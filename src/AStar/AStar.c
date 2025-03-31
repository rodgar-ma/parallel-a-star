#include <stdlib.h>
#include <stdio.h>
#include <string.h>

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
    NodeList prevList = NULL;
    while (tmpList) {
        NodeRecord tmpNode = tmpList->nodes;
        NodeRecord prev = NULL;
        while (tmpNode) {
            if (tmpNode->node == n) {
                if (prev) {
                    prev->next = tmpNode->next;
                } else {
                    tmpList->nodes = tmpNode->next;
                }
            }
            prev = tmpNode;
            tmpNode = tmpNode->next;
        }
        prevList = tmpList;
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

typedef struct {
    int width;
    int height;
    Node ** grid;
} Map;


Map *LoadMap(const char *filename) {
    FILE *file;
    errno_t err;

    if ((err = fopen_s(&file, filename, "r")) != 0) {
        perror("Error abriendo el archivo");
        return NULL;
    }

    int width = 0, height = 0;
    char buffer[1024];

    // Leer dimensiones
    while (fgets(buffer, sizeof(buffer), file)) {
        if (strncmp(buffer, "width", 5) == 0) {
            if ((err = sscanf_s(buffer, "width %d", &width)) != 1) {
                perror("Error abriendo el archivo");
                return NULL;
            }
        } else if (strncmp(buffer, "height", 6) == 0) {
            if ((err = sscanf_s(buffer, "height %d", &height)) != 1) {
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
    Map *map = malloc(sizeof(Map));
    map->width = width;
    map->height = height;
    map->grid = calloc(height, sizeof(Node *));

    // Leer el mapa
    for (int y = 0; y < height; y++) {
        fgets(buffer, sizeof(buffer), file);
        map->grid[y] = calloc(width, sizeof(Node));

        for (int x = 0; x < width; x++) {
            if (buffer[x] == '.') {
                map->grid[y][x] = CreateNode(0, 0);
            } else {
                map->grid[y][x] = NULL;
            }
        }
    }

    fclose(file);

    // Conectar nodos vecinos
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (map->grid[y][x]) {
                if (x > 0 && map->grid[y][x - 1]) AddNeighbor(map->grid[y][x], map->grid[y][x - 1]);
                if (x < width - 1 && map->grid[y][x + 1]) AddNeighbor(map->grid[y][x], map->grid[y][x + 1]);
                if (y > 0 && map->grid[y - 1][x]) AddNeighbor(map->grid[y][x], map->grid[y - 1][x]);
                if (y < height - 1 && map->grid[y + 1][x]) AddNeighbor(map->grid[y][x], map->grid[y + 1][x]);
            }
        }
    }

    return map;
}

void FreeMap(Map *map) {
    for (int y = 0; y < map->height; y++) {
        for (int x = 0; x < map->width; x++) {
            if (map->grid[y][x]) free(map->grid[y][x]);
        }
        free(map->grid[y]);
    }
    free(map->grid);
    free(map);
}

Node GetNode(Map map, int x, int y) {
    if (map.grid[x][y]) return map.grid[x][y];
    return NULL;
}

/*******************************************************************/

int main(int argc, char const *argv[])
{
    printf("\n");
    char* filename = "./maze-map/maze512-1-0.map";
    Map* map = LoadMap(filename);
    
    Node source = GetNode(*map, 1, 1);
    Node goal = GetNode(*map, 511, 511);

    FreeMap(map);

    return 0;
}

