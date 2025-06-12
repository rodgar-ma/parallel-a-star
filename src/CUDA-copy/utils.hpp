#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

using std::cin;
using std::cout;
using std::flush;
using std::min;
using std::max;
using std::endl;
using std::pair;
using std::runtime_error;
using std::string;
using std::vector;
using std::make_pair;
using std::numeric_limits;

const float SQRT2 = 1.4142135623731f;
const int DX[8] = { 1, -1,  0,  0,  1,  1, -1, -1, };
const int DY[8] = { 0,  0,  1, -1,  1, -1,  1, -1, };
const float COST[8] = { 1, 1, 1, 1, SQRT2, SQRT2, SQRT2, SQRT2, };

#endif