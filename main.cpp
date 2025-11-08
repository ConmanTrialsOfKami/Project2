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

// read csv file and turn rows into DataRow objects
vector<DataRow> readCSV(string filename) {
    ifstream file(filename);                // open file
    vector<DataRow> data;                   // holds all rows
    string line;
    getline(file, line);                    // skip header line

    // loop through each line
    while (getline(file, line)) {
        stringstream ss(line);              // break line into cells
        string cell;
        DataRow row;                        // store features + label
        int col = 0;
        
        // go through each comma-separated cell
        while (getline(ss, cell, ',')) {
            // if empty or "NA" → make it 0
            if (cell == "" || cell == "NA"){
                cell = "0";
            }
            
            // change true/false strings to numbers
            if (cell == "TRUE" || cell == "True" || cell == "true"){
                cell = "1";
            }
            if (cell == "FALSE" || cell == "False" || cell == "false"){
                cell = "0";
            }

            double val = atof(cell.c_str());    // convert text → number
            
            // last column is label, others are features
            if (col == 20){
                row.label = (int)val;           // set label
            }
            else {
                row.features.push_back(val);    // add feature value
            }
            col++;                              // move to next column
        }
        data.push_back(row);                    // add this row to dataset
    }
    return data;                                // send back full data
}

// split dataset 80% train / 20% test
void splitData(vector<DataRow>& all, vector<DataRow>& train, vector<DataRow>& test) {
    int splitPoint = all.size() * 0.8;          // where to cut
    for (int i = 0; i < all.size(); i++) {
        if (i < splitPoint) {
            train.push_back(all[i]);            // goes into training set
        }
        else {
            test.push_back(all[i]);             // rest into testing
        }
    }
}

// simple accuracy checker
double accuracy(vector<int> preds, vector<DataRow>& test) {
    int correct = 0;
    for (int i = 0; i < preds.size(); i++){
        if (preds[i] == test[i].label) {        // if guess matches label
            correct++;
        }
    }
    return 100.0 * correct / preds.size();      // percent accuracy
}

int main() {
    // read the dataset from file
    vector<DataRow> data = readCSV("300k.csv");
    if (data.empty()) {
        cout << "Error: couldn't load dataset.\n";
        return 1;                               // stop if load failed
    }

    // split into train and test sets
    vector<DataRow> train;
    vector<DataRow> test;
    splitData(data, train, test);

    // keep looping so user can test different options
    while (true) {
        cout << "Predict 'em All: Water-Proximity\n";
        cout << "1. Decision Tree\n2. Naive Bayes\n3. Compare Both\n4. Exit\n";
        cout << "Choose: ";

        int choice;
        cin >> choice;                          // get user input

        if (choice == 1) {
            // Decision Tree
            clock_t start = clock();            // start timer
            DecisionTree tree;
            tree.train(train);                  // train model
            vector<int> preds = tree.predictAll(test);      // predict
            clock_t end = clock();              // stop timer
            cout << "\nDecision Tree Accuracy: " << accuracy(preds, test) << "%";
            cout << "\nRuntime: " << double(end - start) / CLOCKS_PER_SEC << " sec\n";
        }
        else if (choice == 2) {
            // Naive Bayes
            clock_t start = clock();
            NaiveBayes nb;
            nb.train(train);                    // train NB model
            vector<int> preds = nb.predictAll(test);
            clock_t end = clock();
            cout << "\nNaive Bayes Accuracy: " << accuracy(preds, test) << "%";
            cout << "\nRuntime: " << double(end - start) / CLOCKS_PER_SEC << " sec\n";
        }
        else if (choice == 3) {
            // compare both models on same test
            DecisionTree tree; tree.train(train);
            NaiveBayes nb; nb.train(train);

            vector<int> preds1 = tree.predictAll(test);
            vector<int> preds2 = nb.predictAll(test);

            cout << "\nDecision Tree Accuracy: " << accuracy(preds1, test) << "%";
            cout << "\nNaive Bayes Accuracy: " << accuracy(preds2, test) << "%\n";
        }
        else if (choice == 4) {
            // quit the program
            cout << "Goodbye!\n";
            break;
        }
        else {
            cout << "Invalid choice.\n";        // bad menu input
        }
        cout << "\nPress Enter to exit...";
        cin.ignore();                           // clear buffer
        cin.get();                              // pause
    }
    return 0;
}

