#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
using namespace std;

struct DataRow {
    vector<double> features;
    int label;
};

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
            double val = stod(cell);
            if (col == 5)
                row.label = (int)val;
            else
                row.features.push_back(val);
            col++;
        }
        data.push_back(row);
    }
    return data;
}

int main() {
    vector<DataRow> data = readCSV("pokemon.csv");
    cout << "Loaded " << data.size() << " rows." << endl;
    return 0;
}
