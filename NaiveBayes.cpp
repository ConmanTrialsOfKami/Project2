#include <vector>
#include <cmath>
#include <iostream>
#include "DataRow.h"
using namespace std;


class NaiveBayes {
public:
    vector<double> mean0; // mean values for class 0
    vector<double> mean1; // mean values for class 1
    vector<double> var0; // variances for class 0
    vector<double> var1; // variances for 1
    double prior0; // prior prob of 0
    double prior1; // prior prob of 1

    void train(vector<DataRow>& train) {
        int nFeat = train[0].features.size(); // number of features per row
        mean0.assign(nFeat, 0);
        mean1.assign(nFeat, 0);
        var0.assign(nFeat, 0); // count 0 and 1 samples
        var1.assign(nFeat, 0);

        int x = 0;
        int y = 0;
        // loop through each row
        for (auto& r : train) {
            if (r.label == 0) {
                x++;
                for (int i = 0; i < nFeat; i++) {
                    mean0[i] += r.features[i]; // sum up features for class 0
                }
            } 
            else {
                y++;
                for (int i = 0; i < nFeat; i++) {
                    mean1[i] += r.features[i]; // sum up features for class 1
                }
            }
        }
        // divide by how many means in each class to get average
        for (int i = 0; i < nFeat; i++) {
            mean0[i] /= max(1, x);
            mean1[i] /= max(1, y);
        }
        // calculate variance for each feature per class
        for (auto& r : train) {
            if (r.label == 0){
                for (int i = 0; i < nFeat; i++){
                    var0[i] += pow(r.features[i] - mean0[i], 2);
                }
            }
            else{
                for (int i = 0; i < nFeat; i++){
                    var1[i] += pow(r.features[i] - mean1[i], 2);
                }
            }
        }
        // finish variance calc, add small constant to avoid
        for (int i = 0; i < nFeat; i++) {
            var0[i] = var0[i] / max(1, x) + 1e-6;
            var1[i] = var1[i] / max(1, y) + 1e-6;
        }
        // calculate prior probability of each class
        prior0 = (double)x / train.size() + 1e-6;
        prior1 = (double)y / train.size() + 1e-6;
    }

    int predictOne(DataRow& r) {
        // start with log of prior probabilities
        double logP0 = log(prior0);
        double logP1 = log(prior1);
        // gaussian likelihood without normalization constant
        for (int i = 0; i < r.features.size(); i++) {
            double x = r.features[i];
            logP0 += -pow(x - mean0[i], 2) / (2 * var0[i]);
            logP1 += -pow(x - mean1[i], 2) / (2 * var1[i]);
        }
        // whicever has higher probability wins
        if(logP1 > logP0){
            return 1;
        } else{
            return 0;
        }
    }

    vector<int> predictAll(vector<DataRow>& test) {
        vector<int> preds;
        // predict each row using predictOne
        for (auto& r : test){
            preds.push_back(predictOne(r));
        }
        return preds; // return all predictions
    }
};
