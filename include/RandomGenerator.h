#ifndef RANDOMGENERATOR_H
#define RANDOMGENERATOR_H
#include<random>
#include<cmath>



class RandomGenerator {
protected:
    int fan_in;
    int fan_out;
public:
    RandomGenerator(int fan_in, int fan_out);
    virtual float generate_random_weight() = 0;
    virtual ~RandomGenerator() = default;
};

class Xavier : public RandomGenerator {
private:
    std::mt19937 gen;

public:
    Xavier(int fan_in, int fan_out);

    float generate_random_weight() override;
};

class He : public RandomGenerator {
private:
    std::mt19937 gen;

public:
    He(int fan_in, int fan_out);

    float generate_random_weight() override;
};

#endif // RANDOMGENERATOR_H
