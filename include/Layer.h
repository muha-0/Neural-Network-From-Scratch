#ifndef LAYER_H
#define LAYER_H
#include"Unit.h"
#include<string>
#include<vector>



class Layer{
protected:
    int units_num;
    std::string activation;
    std::vector<Unit> units;
    int input_dim;
    RandomGenerator* decide_generator(const std::string &activation, const int &fan_in, const int &fan_out);

public:
    int get_units_num();
    std::string get_activation();
    int get_input_dim();
    virtual void backward(const std::vector<float> &gradients, const float& alpha) = 0;
    virtual std::vector<float> forward(const std::vector<float>& input) = 0;
    virtual void build(const int &input_dim) = 0;
};

class Dense : public Layer{
private:

public:
    Dense(const int& units_num, const std::string& activation, const int& input_dim = -1);
    std::vector<float> forward(const std::vector<float> &input) override;
    void build(const int &input_dim) override;
    void backward(const std::vector<float> &gradients, const float& alpha) override;
};

#endif // LAYER_H
