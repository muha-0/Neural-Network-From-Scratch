#include "RandomGenerator.h"


RandomGenerator::RandomGenerator(int fan_in, int fan_out) : fan_in(fan_in), fan_out(fan_out) {}






Xavier::Xavier(int fan_in, int fan_out) : RandomGenerator(fan_in, fan_out) {
    std::random_device rd;
    gen = std::mt19937(rd());
}

float Xavier::generate_random_weight(){
    float limit = sqrt(6.0f / (fan_in + fan_out));
    std::uniform_real_distribution<float> dis(-limit, limit);
    return dis(gen);
}




He::He(int fan_in, int fan_out) : RandomGenerator(fan_in, fan_out) {
    std::random_device rd;
    gen = std::mt19937(rd());
}

float He::generate_random_weight(){
    float stddev = sqrt(2.0f / fan_in);
    std::normal_distribution<float> dis(0.0f, stddev);
    return dis(gen);
}
