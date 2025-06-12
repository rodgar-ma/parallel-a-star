#include "astar_seq.hpp"

struct node_t {
    int id;
    float dist;
    node_t *parent;
    node_t() = default;
    node_t(int id, float dist, node_t *parent) : id(id), dist(dist), parent(parent) {}
};

AStarSeq::AStarSeq(AStarSource *source) : source(source) {};

AStarSeq::~AStarSeq() {
    for (auto &pair : globalList)
        delete pair.second;

    globalList.clear();
}

void AStarSeq::initialize() {
    for (auto &pair : globalList)
        delete pair.second;
    globalList.clear();

    openList = decltype(openList)();
    closedList.clear();

    targetID = source->toID(source->ex(), source->ey());

    int startID = source->toID(source->sx(), source->sy());
    node_t *startNode = new node_t(startID, 0, nullprt);
}

bool AStarSeq::solve() {

}

void AStarSeq::getSolution(float *optimal, vector<int> *pathList) {

}