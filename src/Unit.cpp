#include "Unit.h"
#include<vector>
#include<string>
#include <numeric>

Unit::Unit(const int &params_num,RandomGenerator* generator){
    for(int i = 0;i<params_num;i++){
        weights.push_back(generator->generate_random_weight());
    }
    bias = generator->generate_random_weight();
}
float Unit::output(const std::vector<float>& input, const std::string &activation) {
    float sum = bias;
    for (int i = 0; i < weights.size(); ++i) {
        sum += input[i] * weights[i];
    }
    if(activation == "relu"){
        return std::max(0.0f,sum);
    }
    else if(activation == "sigmoid"){
        return 1.0f / (1.0f + exp(-sum));
    }

    //No activation (linear)
    return sum;
}
void Unit::backward(const std::vector<float>& gradients, const float &alpha){

    for(int i = 0;i<weights.size();i++){
        weights[i]-=alpha*gradients[i];
    }

    bias -= alpha * accumulate(gradients.begin(), gradients.end(), 0.0f);

}

