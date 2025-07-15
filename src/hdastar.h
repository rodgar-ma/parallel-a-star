#ifndef HDASTAR_H
#define HDASTAR_H

#include "astar.h"

path *hdastar_search(AStarSource *source, int s_id, int t_id, int k, double *cpu_time_used);

#endif