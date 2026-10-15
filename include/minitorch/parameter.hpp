#pragma once
#include "common.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

/* defines torch-style Parameter structs
    value: the tensor/vector contained in the param (i.e the 'data')
    grad: the gradient 
    m, v: are are fields for the moments for adaptive moment computations
    required by Adam and SGD with Nesterov 
*/

namespace minitorch {

    struct Parameter {
        std::vector<float> value, grad, m, v;
        explicit Parameter(std::size_t n)
        : value(n, 0), grad(n, 0), m(n, 0), v(n, 0) {
        }

        void zero_grad() {
            std::fill(grad.begin(), grad.end(), 0);
        }

        void initialize(std::size_t fan_in, unsigned seed) {
            std::mt19937 rng(seed);
            std::normal_distribution<float> dist(0, std::sqrt(2.f/static_cast<float>(fan_in)));
            for (auto& x :value) x = dist(rng);
        }
    };

}
// namespace minitorch
