#include <vector>
#include <iostream>
#include <cmath>
using namespace std;

struct DataRow {
    vector<double> features;
    int label;
};

class DecisionTree {
public:
    int bestFeature;
    double bestThreshold;
    int lowPred;
    int highPred;

    void train(vector<DataRow>& train) {
        int nFeat = train[0].features.size();
        bestFeature = 0;
        bestThreshold = 0;
        double bestAcc = 0;

        for (int f = 0; f < nFeat; f++) {
            double avg = 0;
            for (auto& r : train){
                avg += r.features[f];
            }
            avg /= train.size();

            int correct = 0;
            for (auto& r : train) {
                int pred;
                if (r.features[f] >= avg) {
                    pred = 1;
                } 
                else {
                    pred = 0;
                }
                if (pred == r.label){
                    correct++;
                }
            }
            double acc = (double)correct / train.size();
            if (acc > bestAcc) {
                bestAcc = acc;
                bestFeature = f;
                bestThreshold = avg;
            }
        }

        int lowCount = 0, highCount = 0;
        int lowSize = 0, highSize = 0;
        for (auto& r : train) {
            if (r.features[bestFeature] < bestThreshold) {
                lowCount += r.label;
                lowSize++;
            } else {
                highCount += r.label;
                highSize++;
            }
        }
        if (lowCount > lowSize/2) {
            lowPred = 1;
        } 
        else {
            lowPred = 0;
        }

        if (highCount > highSize/2) {
            highPred = 1;
        } 
        else {
            highPred = 0;
        }
    }

    int predictOne(DataRow& r) {
        if (r.features[bestFeature] < bestThreshold){
            return lowPred;
        }
        else {
            return highPred;
        }
    }

    vector<int> predictAll(vector<DataRow>& test) {
        vector<int> preds;
        for (auto& r : test){
            preds.push_back(predictOne(r));
        }
        return preds;
    }
};