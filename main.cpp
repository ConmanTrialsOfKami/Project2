#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <ctime>
#include "DecisionTree.cpp"
#include "NaiveBayes.cpp"
#include "DataRow.h"
using namespace std;


vector<DataRow> readCSV(string filename) {
    ifstream file(filename);
    vector<DataRow> data;
    string line;
    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        DataRow row;
        int col = 0;

        while (getline(ss, cell, ',')) {
            if (cell == "" || cell == "NA"){
                cell = "0";
            }

            if (cell == "TRUE" || cell == "True" || cell == "true"){
                cell = "1";
            }
            if (cell == "FALSE" || cell == "False" || cell == "false"){
                cell = "0";
            }

            double val = atof(cell.c_str());
            if (col == 20){
                row.label = (int)val;
            }
            else {
                row.features.push_back(val);
            }
            col++;
        }
        data.push_back(row);
    }
    return data;
}

void splitData(vector<DataRow>& all, vector<DataRow>& train, vector<DataRow>& test) {
    int splitPoint = all.size() * 0.8;
    for (int i = 0; i < all.size(); i++) {
        if (i < splitPoint) {
            train.push_back(all[i]);
        }
        else {
            test.push_back(all[i]);
        }
    }
}

double accuracy(vector<int> preds, vector<DataRow>& test) {
    int correct = 0;
    for (int i = 0; i < preds.size(); i++){
        if (preds[i] == test[i].label) {
            correct++;
        }
    }
    return 100.0 * correct / preds.size();
}

int main() {
    vector<DataRow> data = readCSV("300k.csv");
    if (data.empty()) {
        cout << "Error: couldn't load dataset.\n";
        return 1;
    }

    vector<DataRow> train;
    vector<DataRow> test;
    splitData(data, train, test);
    while (true) {
        cout << "Predict 'em All: Water-Proximity\n";
        cout << "1. Decision Tree\n2. Naive Bayes\n3. Compare Both\n4. Exit\n";
        cout << "Choose: ";

        int choice;
        cin >> choice;

        if (choice == 1) {
            clock_t start = clock();
            DecisionTree tree;
            tree.train(train);
            vector<int> preds = tree.predictAll(test);
            clock_t end = clock();
            cout << "\nDecision Tree Accuracy: " << accuracy(preds, test) << "%";
            cout << "\nRuntime: " << double(end - start) / CLOCKS_PER_SEC << " sec\n";
        }
        else if (choice == 2) {
            clock_t start = clock();
            NaiveBayes nb;
            nb.train(train);
            vector<int> preds = nb.predictAll(test);
            clock_t end = clock();
            cout << "\nNaive Bayes Accuracy: " << accuracy(preds, test) << "%";
            cout << "\nRuntime: " << double(end - start) / CLOCKS_PER_SEC << " sec\n";
        }
        else if (choice == 3) {
            DecisionTree tree; tree.train(train);
            NaiveBayes nb; nb.train(train);

            vector<int> preds1 = tree.predictAll(test);
            vector<int> preds2 = nb.predictAll(test);

            cout << "\nDecision Tree Accuracy: " << accuracy(preds1, test) << "%";
            cout << "\nNaive Bayes Accuracy: " << accuracy(preds2, test) << "%\n";
        }
        else if (choice == 4) {
            cout << "Goodbye!\n";
            break;
        }
        else {
            cout << "Invalid choice.\n";
        }
        cout << "\nPress Enter to exit...";
        cin.ignore();
        cin.get();
    }
    return 0;
}

