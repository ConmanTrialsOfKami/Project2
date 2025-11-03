#ifndef DATAROW_H
#define DATAROW_H
#include <vector>
using namespace std;

struct DataRow {
    vector<double> features;
    int label;
};

#endif