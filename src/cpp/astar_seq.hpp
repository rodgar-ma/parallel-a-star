#ifndef ASTAR_SEQ_HPP
#define ASTAR_SEQ_HPP

#include "astar_source.hpp"
#include <queue>
#include <unordered_set>
#include <unordered_map>

class node_t;

class AStarSeq {
public:
    AStarSeq(AStarSource *source);
    ~AStarSeq();
    void initialize();
    bool solve();
    void getSolution(float *optimal, vector<int> *pathList);

private:
    float computeFValue(node_t *node);
    AStarSource *source;
    std::priority_queue<
        pair<float, node_t *>,
        vector<pair<float, node_t *>>,
        std::greater<pair<float, node_t *>>> openList;
    std::unordered_set<int> closedList;
    std::unordered_map<int, node_t *> globalList;

    int targetID;
    node_t *optimalNode;
};


#endif
