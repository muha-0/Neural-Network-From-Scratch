#include "Model.h"
#include"Layer.h"
#include<vector>
#include<string>
#include<iostream>
std::vector<float> Model::forward(std::vector<float> input) {
    for (auto &layer : layers) {
        input = layer->forward(input);
    }
    return input;
}
float Model::compute_cost(const std::vector< std::vector<float> > &outputs, const std::vector<float> &y){
    //For now just MSE
    int m = outputs.size();
    float total_loss = 0.0f;
    for(int i = 0;i<m;i++){
        float error = outputs[i][0] - y[i]; //assuming 1 output
        total_loss += error*error;
    }
    return total_loss/(2*m);
}
void Model::backward_prob(const std::vector<std::vector<float>>& X, const std::vector<float>& y) {
    float alpha = 0.001;
    //For simplicity we assume one layer in the neural network or in other words we update all according to the label only
    std::vector<float> gradients(X[0].size());
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

Model::Model(const std::vector<Layer*> &layers){
    this->layers = layers;
}
void Model::compile(const std::string &loss_function, const std::string &optimizer){
    this->loss_function = loss_function;
    this->optimizer = optimizer;
    for (int i = 0; i < layers.size(); ++i) {
        int input_dim = (i == 0) ? layers[0]->get_input_dim() : layers[i - 1]->get_units_num();
        layers[i]->build(input_dim);
    }
}
//Assuming batch size equals the dataset size always
void Model::fit(const std::vector< std::vector<float> > &X, const std::vector<float> &y, const int &epochs){
    int traning_examples = X.size();
    float cost;
    std::vector < std::vector <float> > outputs;
    for(int i = 0;i<epochs;i++){
        outputs.clear();
        //Forward Probagation
        for(int j = 0;j<traning_examples;j++){
            outputs.push_back(forward(X[j]));
        }
        cost = compute_cost(outputs,y);
        std::cout << "Epoch " << i + 1 << ", Cost: " << cost << std::endl;

        backward_prob(X,y);

    }
}
float Model::predict(const std::vector<float> &x){
    std::vector<float> y = forward(x);
    //Assuming y.size() == 1 for now
    return y[0];
}
