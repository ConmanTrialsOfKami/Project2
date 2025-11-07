#include <vector>
#include <iostream>
#include <cmath>
#include "DataRow.h"
using namespace std;



class DecisionTree {
public:
    int bestFeature; // which feature gives best split
    double bestThreshold; // value to split on
    int lowPred; // prediction if below threshold
    int highPred; // prediction for above threshold

    void train(vector<DataRow>& train) {
        int nFeat = train[0].features.size(); // number of features
        bestFeature = 0;
        bestThreshold = 0;
        double bestAcc = 0; // best accuracy
        // loop thru each  feature
        for (int f = 0; f < nFeat; f++) {
            double avg = 0; // mean of this feature
            for (auto& r : train){
                avg += r.features[f]; // add up all the feature values
            }
            avg /= train.size(); // avg threshold

            int correct = 0; // how many we predict right
            for (auto& r : train) {
                int pred;
                if (r.features[f] >= avg) { // if value >= avg predict 1
                    pred = 1;
                } 
                else {
                    pred = 0; // else predict 0
                }
                if (pred == r.label){ // if match label, count it
                    correct++;
                }
            }
            double acc = (double)correct / train.size(); // accuracy
            if (acc > bestAcc) { // keep if better split
                bestAcc = acc;
                bestFeature = f;
                bestThreshold = avg;
            }
        }
        // cout how many 1s and 0s on both sides of split
        int lowCount = 0, highCount = 0;
        int lowSize = 0, highSize = 0;
        for (auto& r : train) {
            if (r.features[bestFeature] < bestThreshold) {
                lowCount += r.label; // add label to low side
                lowSize++;
            } else {
                highCount += r.label;
                highSize++;
            }
        }

        // majority vote for each side
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
        // chec which side of threshold it's one
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
            preds.push_back(predictOne(r)); //predict for each row
        }
        return preds; // return all predictions
    }
};