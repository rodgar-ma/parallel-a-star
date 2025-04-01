#include <stdlib.h>
#include "AStar.h"

// Priority queue implementation for A*
typedef struct {
    Node node;
    float priority;
} PQElement;

typedef struct {
    int capacity;
    int count;
    PQElement* first;
} PriorityQueue;

PriorityQueue* pq_create(int capacity) {
    PriorityQueue* pq = malloc(sizeof(PriorityQueue));
    pq->first = malloc(sizeof(PQElement) * capacity);
    pq->capacity = capacity;
    pq->count = 0;
    return pq;
}

void pq_free(PriorityQueue* pq) {
    free(pq->first);
    free(pq);
}

void pq_push(PriorityQueue* pq, Node node, float priority) {
    if (pq->size == pq->capacity) {
        pq->capacity *= 2;
        pq->elements = realloc(pq->elements, sizeof(PQElement) * pq->capacity);
    }
    
    int i = pq->size++;
    pq->elements[i].node = node;
    pq->elements[i].priority = priority;
    
    // Bubble up
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (pq->elements[i].priority >= pq->elements[parent].priority) break;
        PQElement temp = pq->elements[i];
        pq->elements[i] = pq->elements[parent];
        pq->elements[parent] = temp;
        i = parent;
    }
}

Node pq_pop(PriorityQueue* pq) {
    if (pq->size == 0) return NULL;
    
    Node result = pq->elements[0].node;
    pq->elements[0] = pq->elements[--pq->size];
    
    // Bubble down
    int i = 0;
    while (1) {
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        int smallest = i;
        
        if (left < pq->size && pq->elements[left].priority < pq->elements[smallest].priority)
            smallest = left;
        if (right < pq->size && pq->elements[right].priority < pq->elements[smallest].priority)
            smallest = right;
        if (smallest == i) break;
        
        PQElement temp = pq->elements[i];
        pq->elements[i] = pq->elements[smallest];
        pq->elements[smallest] = temp;
        i = smallest;
    }
    
    return result;
}

int pq_empty(PriorityQueue* pq) {
    return pq->count == 0;
}

// A* algorithm implementation
typedef struct {
    Node node;
    Node came_from;
    float g_score;
    float f_score;
} NodeRecord;

typedef struct {
    AStarData* data;
    int capacity;
    int size;
} AStarDataMap;

AStarDataMap* as_datamap_create(int capacity) {
    AStarDataMap* map = malloc(sizeof(AStarDataMap));
    map->data = malloc(sizeof(AStarData) * capacity);
    map->capacity = capacity;
    map->size = 0;
    return map;
}

void as_datamap_free(AStarDataMap* map) {
    free(map->data);
    free(map);
}

AStarData* as_datamap_get(AStarDataMap* map, Node node, CompareFunc compare) {
    for (int i = 0; i < map->size; i++) {
        if (compare(map->data[i].node, node)) {
            return &map->data[i];
        }
    }
    return NULL;
}

void as_datamap_add(AStarDataMap* map, Node node, Node came_from, float g_score, float f_score) {
    if (map->size == map->capacity) {
        map->capacity *= 2;
        map->data = realloc(map->data, sizeof(AStarData) * map->capacity);
    }
    
    map->data[map->size].node = node;
    map->data[map->size].came_from = came_from;
    map->data[map->size].g_score = g_score;
    map->data[map->size].f_score = f_score;
    map->size++;
}

// Reconstruct the path from the came_from map
Node* reconstruct_path(AStarDataMap* came_from, Node start, Node goal, CompareFunc compare, int* path_length) {
    Node* path = NULL;
    *path_length = 0;
    
    Node current = goal;
    while (current != NULL && !compare(current, start)) {
        path = realloc(path, sizeof(Node) * (*path_length + 1));
        path[*path_length] = current;
        (*path_length)++;
        
        AStarData* data = as_datamap_get(came_from, current, compare);
        current = data ? data->came_from : NULL;
    }
    
    // Add start node
    path = realloc(path, sizeof(Node) * (*path_length + 1));
    path[*path_length] = start;
    (*path_length)++;
    
    // Reverse the path
    for (int i = 0; i < *path_length / 2; i++) {
        Node temp = path[i];
        path[i] = path[*path_length - i - 1];
        path[*path_length - i - 1] = temp;
    }
    
    return path;
}

// Main A* function
Node* a_star(AStarSource *source, Node start, Node goal) {
    
    PriorityQueue* open_set = pq_create(100);
    AStarDataMap* came_from = as_datamap_create(100);
    AStarDataMap* g_scores = as_datamap_create(100);
    AStarDataMap* f_scores = as_datamap_create(100);
    
    // Initialize with start node
    float h = heuristic(start, goal);
    pq_push(open_set, start, h);
    as_datamap_add(g_scores, start, NULL, 0.0f, 0.0f);
    as_datamap_add(f_scores, start, NULL, 0.0f, h);
    
    while (!pq_empty(open_set)) {
        Node current = pq_pop(open_set);
        
        if (compare(current, goal)) {
            // Path found
            Node* path = reconstruct_path(came_from, start, goal, compare, path_length);
            
            // Clean up
            pq_free(open_set);
            as_datamap_free(came_from);
            as_datamap_free(g_scores);
            as_datamap_free(f_scores);
            
            return path;
        }
        
        int neighbor_count = 0;
        Node* neighbors = get_neighbors(current, &neighbor_count);
        
        for (int i = 0; i < neighbor_count; i++) {
            Node neighbor = neighbors[i];
            
            // Get current g_score for the neighbor
            AStarData* current_g_data = as_datamap_get(g_scores, current, compare);
            float tentative_g_score = current_g_data->g_score + cost(current, neighbor);
            
            AStarData* neighbor_g_data = as_datamap_get(g_scores, neighbor, compare);
            
            if (neighbor_g_data == NULL || tentative_g_score < neighbor_g_data->g_score) {
                // This path to neighbor is better than any previous one
                as_datamap_add(came_from, neighbor, current, 0.0f, 0.0f);
                as_datamap_add(g_scores, neighbor, NULL, tentative_g_score, 0.0f);
                
                float f_score = tentative_g_score + heuristic(neighbor, goal);
                as_datamap_add(f_scores, neighbor, NULL, 0.0f, f_score);
                
                // Check if neighbor is already in open set
                bool in_open_set = false;
                for (int j = 0; j < open_set->size; j++) {
                    if (compare(open_set->elements[j].node, neighbor)) {
                        in_open_set = true;
                        open_set->elements[j].priority = f_score;
                        break;
                    }
                }
                
                if (!in_open_set) {
                    pq_push(open_set, neighbor, f_score);
                }
            }
        }
        
        free(neighbors);
    }
    
    // No path found
    pq_free(open_set);
    as_datamap_free(came_from);
    as_datamap_free(g_scores);
    as_datamap_free(f_scores);
    
    *path_length = 0;
    return NULL;
}