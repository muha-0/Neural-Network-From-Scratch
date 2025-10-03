#include<bits/stdc++.h>
#include<iostream>
#include <fstream>

#include"Unit.h"
#include"Layer.h"
#include"Model.h"
#include"RandomGenerator.h"
using namespace std;

// Helper functions to compute mean of a vector
float compute_mean(const vector<float>& data) {
    float sum = 0.0f;
    for (float val : data) {
        sum += val;
    }
    return sum / data.size();
}

// Helper function to compute standard deviation
float compute_stddev(const vector<float>& data, float mean) {
    float variance = 0.0f;
    for (float val : data) {
        variance += (val - mean) * (val - mean);
    }
    variance /= data.size();  // Population stddev (use data.size() - 1 for sample stddev)
    return sqrt(variance);
}

// Normalize all features column-wise
void normalize_features(vector<vector<float>>& X) {
    if (X.empty()) return;

    // Transpose the matrix to work with columns
    vector<vector<float>> columns(X[0].size(), vector<float>(X.size()));
    for (size_t i = 0; i < X.size(); ++i) {
        for (size_t j = 0; j < X[i].size(); ++j) {
            columns[j][i] = X[i][j];
        }
    }

    // Normalize each column
    for (auto& col : columns) {
        float col_mean = compute_mean(col);
        float col_stddev = compute_stddev(col, col_mean);

        // Handle constant features (stddev = 0)
        if (col_stddev < 1e-7) {
            col_stddev = 1.0f;  // Prevent division by zero
        }

        for (float& val : col) {
            val = (val - col_mean) / col_stddev;
        }
    }

    // Transpose back to original format
    X = vector<vector<float>>(columns[0].size(), vector<float>(columns.size()));
    for (size_t i = 0; i < columns.size(); ++i) {
        for (size_t j = 0; j < columns[i].size(); ++j) {
            X[j][i] = columns[i][j];
        }
    }
}

int main() {

    ifstream file("Housing.csv");  // path to your CSV file
    string line;
    vector< vector<string> > X_read;
    vector<string> y_read;
    bool first_row = true;
    while (getline(file, line)) {
        if(first_row){
            first_row = false;
            continue;
        }
        stringstream ss(line);
        string cell;
        vector<string> row;
        //The first column is the target variable
        bool first_column = true;
        while (getline(ss, cell, ',')) {
            if(first_column){
                y_read.push_back(cell);
                first_column = false;
            }
            else{
                row.push_back(cell);
            }
        }

        // Print the row
        /*for (const string& val : row) {
            cout << val << " ";
        }
        cout<<endl;*/
        X_read.push_back(row);

    }

    int sz = X_read.size();

    int row_sz = X_read[0].size();

    vector< vector<float> > X;
    vector<float> y;

    for(int i = 0;i<sz;i++){
        vector<float> row;
        for(int j = 0;j<row_sz;j++){
            switch (j) {

                case 0:
                case 1:
                case 2:
                case 3:
                case 9:

                    row.push_back(stof(X_read[i][j]));
                    break;

                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 10:
                    row.push_back(X_read[i][j] == "yes" ? 1 : 0);
                    break;

                case 11:
                    if (X_read[i][j] == "furnished")
                        row.push_back(2);
                    else if (X_read[i][j] == "semi-furnished")
                        row.push_back(1);
                    else
                        row.push_back(0);
                    break;
            }

        }
        y.push_back(stof(y_read[i]));
        X.push_back(row);
    }

    // Normalize features
    normalize_features(X);

    // Also normalize target prices (y)
    float y_mean = compute_mean(y);
    float y_stddev = compute_stddev(y, y_mean);
    for (float& price : y) {
        price = (price - y_mean) / y_stddev;
    }

    Model model = Model({
        new Dense(1,"relu",X[0].size())
    });


    model.compile("MSE","gradient_descent");

    model.fit(X,y,4000);
    //Test on 300 examples from the training set. This is wrong but I just want to prove that the NN predicts well on the training
    int m_test = 300;
    float abs_error = 0;
    for(int i = 0;i<m_test;i++){
        float normalized_pred = model.predict({X[i]});
        float denormalized_pred = normalized_pred*y_stddev + y_mean;
        float denormalized_y = y[i]*y_stddev + y_mean;
        cout<<"Predicted: "<<denormalized_pred<<endl;
        abs_error += abs(denormalized_pred-denormalized_y);
    }
    cout<<"Average error: "<< abs_error/m_test<<endl;


    file.close();
    return 0;
}
