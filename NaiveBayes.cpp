#include <vector>
#include <cmath>
#include <iostream>
using namespace std;

struct DataRow {
    vector<double> features;
    int label;
};


class NaiveBayes {
public:
    vector<double> mean0;
    vector<double> mean1;
    vector<double> var0;
    vector<double> var1;
    double prior0
    double prior1;

    void train(vector<DataRow>& train) {
        int nFeat = train[0].features.size();
        mean0.assign(nFeat, 0);
        mean1.assign(nFeat, 0);
        var0.assign(nFeat, 0);
        var1.assign(nFeat, 0);

        int x = 0
        int y = 0;

        for (auto& r : train) {
            if (r.label == 0) {
                x++;
                for (int i = 0; i < nFeat; i++) {
                    mean0[i] += r.features[i];
                }
            } 
            else {
                y++;
                for (int i = 0; i < nFeat; i++) {
                    mean1[i] += r.features[i];
                }
            }
        }
        for (int i = 0; i < nFeat; i++) {
            mean0[i] /= x;
            mean1[i] /= y;
        }

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
        for (int i = 0; i < nFeat; i++) {
            var0[i] /= x;
            var1[i] /= y;
        }

        prior0 = (double)x / train.size();
        prior1 = (double)y / train.size();
    }
}
