#ifndef SPASTAR_H
#define SPASTAR_H

#include "astar.h"

path *spastar_search(AStarSource *source, int s_id, int t_id, int k, double *cpu_time_used);

#endif