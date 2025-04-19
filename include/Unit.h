#ifndef UNIT_H
#define UNIT_H
#include<vector>
#include<string>
#include"RandomGenerator.h"


class Unit{
private:
    std::vector<float> weights;
    float bias;
public:
    Unit(const int &params_num,RandomGenerator* generator);
    float output(const std::vector<float>& input, const std::string &activation);
    void backward(const std::vector<float>& gradients, const float &alpha);
};

#endif // UNIT_H
