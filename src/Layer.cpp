#include "Layer.h"
#include "Unit.h"
#include "RandomGenerator.h"
#include <vector>
#include <string>

RandomGenerator* Layer::decide_generator(const std::string &activation, const int &fan_in, const int &fan_out){
    if(activation == "relu"){
        return new He(fan_in, fan_out);
    }
    else{
        return new Xavier(fan_in, fan_out);
    }
}


int Layer::get_units_num(){
    return units_num;
}
std::string Layer::get_activation(){
    return activation;
}
int Layer::get_input_dim(){
    return input_dim;
}




Dense::Dense(const int& units_num, const std::string& activation, const int& input_dim) {
    this->units_num = units_num;
    this->activation = activation;
    this->input_dim = input_dim;
}

std::vector<float> Dense::forward(const std::vector<float> &input){
    std::vector<float> output;
    for(auto& unit : units){
        output.push_back(unit.output(input,activation));
    }
    return output;
}
void Dense::build(const int &input_dim){
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
void Dense::backward(const std::vector<float> &gradients, const float& alpha){
    for(int i = 0;i<units_num;i++){
        units[i].backward(gradients,alpha);
    }
}


