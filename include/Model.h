#ifndef MODEL_H
#define MODEL_H
#include"Layer.h"
#include<vector>
#include<string>

class Model{
private:
    std::vector<Layer*> layers;
    std::string loss_function;
    std::string optimizer;
    std::vector<float> forward(std::vector<float> input);
    float compute_cost(const std::vector< std::vector<float> > &outputs, const std::vector<float> &y);
    void backward_prob(const std::vector<std::vector<float>>& X, const std::vector<float>& y);
public:
    Model(const std::vector<Layer*> &layers);
    void compile(const std::string &loss_function, const std::string &optimizer);
    void fit(const std::vector< std::vector<float> > &X, const std::vector<float> &y, const int &epochs);
    float predict(const std::vector<float> &x);
};

#endif // MODEL_H
