#include "astar.h"
#include "heap.h"
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <stddef.h>

#define TAG_REQ_PARENT 1
#define TAG_RES_PARENT 2

node_t *node_create(int id, float gCost, float fCost, int parent) {
    node_t *n = malloc(sizeof(node_t));
    n->id = id;
    n->parent = parent;
    n->gCost = gCost;
    n->fCost = fCost;
    return n;
}

neighbors_list *neighbors_list_create(void) {
    neighbors_list *list = malloc(sizeof(neighbors_list));
    list->capacity = INIT_NEIGHBORS_LIST_CAPACITY;
    list->count = 0;
    list->costs = malloc(list->capacity * sizeof(float));
    list->nodeIds = malloc(list->capacity * sizeof(int));
    return list;
}

void neighbors_list_destroy(neighbors_list *list) {
    free(list->nodeIds);
    free(list->costs);
    free(list);
}

void add_neighbor(neighbors_list *neighbors, int n_id, float cost) {
    if (neighbors->count == neighbors->capacity) {
        neighbors->capacity *= 2;
        neighbors->nodeIds = realloc(neighbors->nodeIds, neighbors->capacity * sizeof(int));
        neighbors->costs = realloc(neighbors->costs, neighbors->capacity * sizeof(float));
    }
    neighbors->nodeIds[neighbors->count] = n_id;
    neighbors->costs[neighbors->count] = cost;
    neighbors->count++;
}

static inline int hash(int node, int k) {
    return node % k;
}

path *retrace_path_distributed(node_t **closed, int goal_id, int k, int rank) {
    int current = goal_id;
    int capacity = 64;
    int count = 0;
    int *nodes = malloc(capacity * sizeof(int));

    while (current != -1) {
        if (count == capacity) {
            capacity *= 2;
            nodes = realloc(nodes, capacity * sizeof(int));
        }
        nodes[count++] = current;

        int owner = hash(current, k);
        if (rank == owner) {
            current = closed[current] ? closed[current]->parent : -1;
        } else {
            MPI_Send(&current, 1, MPI_INT, owner, TAG_REQ_PARENT, MPI_COMM_WORLD);
            MPI_Recv(&current, 1, MPI_INT, owner, TAG_RES_PARENT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Invertir el camino
    path *p = malloc(sizeof(path));
    p->count = count;
    p->nodeIds = malloc(count * sizeof(int));
    for (int i = 0; i < count; i++) {
        p->nodeIds[i] = nodes[count - i - 1];
    }

    // Obtener el coste total desde el último nodo
    int last_node = p->nodeIds[p->count - 1];
    int owner = hash(last_node, k);
    if (rank == owner) {
        p->cost = closed[last_node]->gCost;
    } else {
        // Pedir el gCost
        float g;
        MPI_Send(&last_node, 1, MPI_INT, owner, TAG_REQ_PARENT + 2, MPI_COMM_WORLD);
        MPI_Recv(&g, 1, MPI_FLOAT, owner, TAG_REQ_PARENT + 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        p->cost = g;
    }

    free(nodes);
    return p;
}

void serve_parent_requests(node_t **closed, int rank, int k) {
    int finished = 0;
    MPI_Status status;

    while (!finished) {
        int flag = 0;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);

        if (flag) {
            if (status.MPI_TAG == TAG_REQ_PARENT) {
                int query_id;
                MPI_Recv(&query_id, 1, MPI_INT, status.MPI_SOURCE, TAG_REQ_PARENT, MPI_COMM_WORLD, &status);
                int parent = closed[query_id] ? closed[query_id]->parent : -1;
                MPI_Send(&parent, 1, MPI_INT, status.MPI_SOURCE, TAG_RES_PARENT, MPI_COMM_WORLD);
            } else if (status.MPI_TAG == TAG_REQ_PARENT + 2) {
                int query_id;
                MPI_Recv(&query_id, 1, MPI_INT, status.MPI_SOURCE, TAG_REQ_PARENT + 2, MPI_COMM_WORLD, &status);
                float g = closed[query_id] ? closed[query_id]->gCost : -1;
                MPI_Send(&g, 1, MPI_FLOAT, status.MPI_SOURCE, TAG_REQ_PARENT + 2, MPI_COMM_WORLD);
            }
        } else {
            // Si no hay mensajes, dormir un poco o terminar si sabes que nadie más va a preguntar
            // Puedes usar un timeout o una señal global
            finished = 1;  // O manejarlo más sofisticadamente
        }
    }
}

void path_destroy(path *p) {
    free(p->nodeIds);
    free(p);
}

void closed_list_destroy(node_t **closed, int size) {
    for(int i = 0; i < size; i++) {
        if (closed[i] != NULL) free(closed[i]);
    }
    free(closed);
}

typedef struct {
    int id;
    float gCost;
    float fCost;
    int parent;
} node_mpi_t;

node_mpi_t node_to_mpi(node_t *n) {
    return (node_mpi_t){n->id, n->gCost, n->fCost, n->parent};
}

node_t *node_from_mpi(node_mpi_t m) {
    node_t *n = malloc(sizeof(node_t));
    n->id = m.id;
    n->parent = m.parent;
    n->gCost = m.gCost;
    n->fCost = m.fCost;
    return n;
}

void create_mpi_node_type(MPI_Datatype *type) {
    int block_lengths[4] = {1, 1, 1, 1};
    MPI_Aint offsets[4];
    MPI_Datatype types[4] = {MPI_INT, MPI_FLOAT, MPI_FLOAT, MPI_INT};
    offsets[0] = offsetof(node_mpi_t, id);
    offsets[1] = offsetof(node_mpi_t, gCost);
    offsets[2] = offsetof(node_mpi_t, fCost);
    offsets[3] = offsetof(node_mpi_t, parent);

    MPI_Type_create_struct(4, block_lengths, offsets, types, type);
    MPI_Type_commit(type);
}

/**********************************************************************************************************/
/*                                              A* Algorithm                                              */
/**********************************************************************************************************/

path *astar_search(AStarSource *source, int start_id, int goal_id, double *cpu_time_used) {

    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    node_t **closed = malloc(source->max_size * sizeof(node_t*));
    memset(closed, 0, source->max_size * sizeof(node_t*));

    heap_t *open = heap_init();
    neighbors_list *neighbors = neighbors_list_create();

    MPI_Datatype MPI_NODE;
    create_mpi_node_type(&MPI_NODE);
    
    int owner = hash(start_id, size);
    if (rank == owner) {
        node_t *n = node_create(start_id, 0, source->heuristic(start_id, goal_id), -1);
        heap_insert(open, n);
        closed[start_id] = n;
    }

    int found = 0;
    int global_found = 0;

    double start = MPI_Wtime();

    while (!global_found) {

        MPI_Status status;
        int flag = 1;
        while (flag) {
            node_mpi_t msg;
            MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
            if (flag) {
                MPI_Recv(&msg, 1, MPI_NODE, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &status);
                if (!closed[msg.id] || msg.gCost < closed[msg.id]->gCost) {
                    if (closed[msg.id]) free(closed[msg.id]);
                    closed[msg.id] = node_from_mpi(msg);
                    heap_insert(open, closed[msg.id]);
                }
            }
        }

        MPI_Allreduce(&found, &global_found, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if (!heap_is_empty(open) && !global_found) {
            node_t *current = heap_extract(open);

            // printf("Hilo %d expande nodo %d\n", rank, current->id);

            if (current->id == goal_id) {
                found = 1;
                continue;
            }

            neighbors->count = 0;
            source->get_neighbors(neighbors, current->id);

            for (int i = 0; i < neighbors->count; i++) {
                int n_id = neighbors->nodeIds[i];
                float new_cost = current->gCost + neighbors->costs[i];
                int owner = hash(n_id, size);
                if (rank == owner) {
                    if (!closed[n_id] || new_cost < closed[n_id]->gCost) {
                        if (closed[n_id]) free(closed[n_id]);
                        closed[n_id] = node_create(n_id, new_cost, new_cost + source->heuristic(n_id, goal_id), current->id);
                        heap_insert(open, closed[n_id]);
                    }
                } else {
                    node_mpi_t msg = {
                        .id = n_id,
                        .gCost = new_cost,
                        .fCost = new_cost + source->heuristic(n_id, goal_id),
                        .parent = current->id
                    };
                    MPI_Send(&msg, 1, MPI_NODE, owner, 0, MPI_COMM_WORLD);
                }
            }
        }
    }

    printf("Camino encontrado\n");

    *cpu_time_used = MPI_Wtime() - start;

    if (rank == 1) printf("Tiempo: %f\n", 10e3 * (*cpu_time_used));

    path *p_local = NULL;
    int path_owner = hash(goal_id, size);

    if (rank == path_owner) {
        p_local = retrace_path_distributed(closed, goal_id, size, rank);
    }

    // Broadcast del path a todos los procesos
    path *p = NULL;
    if (rank == path_owner) {
        for (int i = 0; i < size; i++) {
            if (i != rank) {
                MPI_Send(&p_local->count, 1, MPI_INT, i, 100, MPI_COMM_WORLD);
                MPI_Send(&p_local->cost, 1, MPI_FLOAT, i, 101, MPI_COMM_WORLD);
                MPI_Send(p_local->nodeIds, p_local->count, MPI_INT, i, 102, MPI_COMM_WORLD);
            }
        }
        p = p_local;
    } else {
        int count;
        float cost;
        MPI_Recv(&count, 1, MPI_INT, path_owner, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&cost, 1, MPI_FLOAT, path_owner, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int *nodes = malloc(count * sizeof(int));
        MPI_Recv(nodes, count, MPI_INT, path_owner, 102, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        p = malloc(sizeof(path));
        p->count = count;
        p->cost = cost;
        p->nodeIds = nodes;
    }

    // Limpieza
    neighbors_list_destroy(neighbors);
    heap_destroy(open);
    closed_list_destroy(closed, source->max_size);
    MPI_Type_free(&MPI_NODE);
    MPI_Finalize();
    return p;
}
