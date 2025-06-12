#ifndef ASTAR_SOURCE_HPP
#define ASTAR_SOURCE_HPP

typedef struct {
    int start_x;
    int start_y;
    int target_x;
    int target_y;
    int map_size;
    int map_width;
    int map_height;
    int *graph;
} AStarSource;

void init_source(AStarSource *source, char *filename);

#endif