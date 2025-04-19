#include<bits/stdc++.h>
#include<iostream>
#include <fstream>

#include"Unit.h"
#include"Layer.h"
#include"Model.h"
#include"RandomGenerator.h"
using namespace std;

/*class RandomGenerator {
protected:
    int fan_in;
    int fan_out;

public:
    RandomGenerator(int fan_in, int fan_out) : fan_in(fan_in), fan_out(fan_out) {}
    virtual float generate_random_weight() = 0;
    virtual ~RandomGenerator() = default;
};

class Xavier : public RandomGenerator {
private:
    mt19937 gen;

public:
    Xavier(int fan_in, int fan_out) : RandomGenerator(fan_in, fan_out) {
        random_device rd;
        gen = mt19937(rd());
    }

    float generate_random_weight() override {
        float limit = sqrt(6.0f / (fan_in + fan_out));
        uniform_real_distribution<float> dis(-limit, limit);
        return dis(gen);
    }
};

class He : public RandomGenerator {
private:
    mt19937 gen;

public:
    He(int fan_in, int fan_out) : RandomGenerator(fan_in, fan_out) {
        random_device rd;
        gen = mt19937(rd());
    }

    float generate_random_weight() override {
        float stddev = sqrt(2.0f / fan_in);
        normal_distribution<float> dis(0.0f, stddev);
        return dis(gen);
    }
};

class Unit{
private:
    vector<float> weights;
    float bias;

public:
    Unit(const int &params_num,RandomGenerator* generator){
        for(int i = 0;i<params_num;i++){
            weights.push_back(generator->generate_random_weight());
        }
        bias = generator->generate_random_weight();
    }
    float output(const vector<float>& input, const string &activation) {
        float sum = bias;
        for (int i = 0; i < weights.size(); ++i) {
            sum += input[i] * weights[i];
        }
        if(activation == "relu"){
            return max(0.0f,sum);
        }
        else if(activation == "sigmoid"){
            return 1.0f / (1.0f + exp(-sum));
        }

        //No activation (linear)
        return sum;
    }
    void backward(const vector<float>& gradients, const float &alpha){

        for(int i = 0;i<weights.size();i++){
            weights[i]-=alpha*gradients[i];
        }

        bias -= alpha * accumulate(gradients.begin(), gradients.end(), 0.0f);

    }
};

class Layer{
protected:

    int units_num;
    string activation;
    vector<Unit> units;
    int input_dim;

    RandomGenerator* decide_generator(const string &activation, const int &fan_in, const int &fan_out){
        if(activation == "relu"){
            return new He(fan_in, fan_out);
        }
        else{
            return new Xavier(fan_in, fan_out);
        }
    }

public:
    int get_units_num(){
        return units_num;
    }
    string get_activation(){
        return activation;
    }
    int get_input_dim(){
        return input_dim;
    }
    virtual void backward(const vector<float> &gradients, const float& alpha) = 0;
    virtual vector<float> forward(const vector<float>& input) = 0;
    virtual void build(const int &input_dim) = 0;
};

class Dense : public Layer{
private:

public:
    Dense(const int& units_num, const string& activation, const int& input_dim = -1) {
        this->units_num = units_num;
        this->activation = activation;
        this->input_dim = input_dim;
    }

    vector<float> forward(const vector<float> &input) override{
        vector<float> output;
        for(auto& unit : units){
            output.push_back(unit.output(input,activation));
        }
        return output;
    }
    void build(const int &input_dim) override {
        this->input_dim = input_dim;
        int fan_in = input_dim;
        int fan_out = units_num;

        RandomGenerator* generator = decide_generator(activation, fan_in, fan_out);
        units.clear();
        for (int i = 0; i < units_num; ++i) {
            units.push_back(Unit(input_dim, generator));
        }
        delete generator;
    }
    void backward(const vector<float> &gradients, const float& alpha) override {
        for(int i = 0;i<units_num;i++){
            units[i].backward(gradients,alpha);
        }
    }

};

class Model{
private:
    vector<Layer*> layers;
    string loss_function;
    string optimizer;
    vector<float> forward(vector<float> input) {
        for (auto &layer : layers) {
            input = layer->forward(input);
        }
        return input;
    }
    float compute_cost(const vector< vector<float> > &outputs, const vector<float> &y){
        //For now just MSE
        int m = outputs.size();
        float total_loss = 0.0f;
        for(int i = 0;i<m;i++){
            float error = outputs[i][0] - y[i]; //assuming 1 output
            total_loss += error*error;
        }
        return total_loss/(2*m);
    }
    void backward_prob(const vector<vector<float>>& X, const vector<float>& y) {
        float alpha = 0.001;
        //For simplicity we assume one layer in the neural network or in other words we update all according to the label only
        vector<float> gradients(X[0].size());
        for(int i = 0;i<X.size();i++){
            for(int j = 0;j<X[0].size();j++){
                gradients[j]+= (predict(X[i]) - y[i])*X[i][j];
            }

        }
        //Divide all by m
        for (int j = 0; j < gradients.size(); j++) {
            gradients[j] /= X.size();
        }

        for(int i = 0;i<layers.size();i++){
            layers[i]->backward(gradients,alpha);
        }

    }

public:

    Model(const vector<Layer*> &layers){
        this->layers = layers;
    }
    void compile(const string &loss_function, const string &optimizer){
        this->loss_function = loss_function;
        this->optimizer = optimizer;
        for (int i = 0; i < layers.size(); ++i) {
            int input_dim = (i == 0) ? layers[0]->get_input_dim() : layers[i - 1]->get_units_num();
            layers[i]->build(input_dim);
        }
    }
    //Assuming batch size equals the dataset size always
    void fit(const vector< vector<float> > &X, const vector<float> &y, const int &epochs){
        int traning_examples = X.size();
        float cost;
        vector < vector <float> > outputs;
        for(int i = 0;i<epochs;i++){
            outputs.clear();
            //Forward Probagation
            for(int j = 0;j<traning_examples;j++){
                outputs.push_back(forward(X[j]));
            }
            cost = compute_cost(outputs,y);
            cout << "Epoch " << i + 1 << ", Cost: " << cost << std::endl;

            backward_prob(X,y);

        }
    }
    float predict(const vector<float> &x){
        vector<float> y = forward(x);
        //Assuming y.size() == 1 for now
        return y[0];
    }


};
*/
// Helper function to compute mean of a vector
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
