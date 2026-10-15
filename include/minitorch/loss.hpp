#pragma once
#include "tensor.hpp"
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace minitorch {

    // this is for testing: we wanna run FashionMNIST/MNIST style datasets
    // -> softmax + cross_entropy are all we need
    // for regression problems just add MSE

    inline Tensor softmax(const Tensor& logits) {
        logits.validate();
        require(logits.n>0 && 
                logits.c>0 && 
                logits.d == 1 && 
                logits.h == 1 && 
                logits.w == 1, 
                "Softmax expects nonempty N,C logits"
                );
        Tensor p = logits;
        for (std::size_t b = 0; b<p.n; ++b) {

            auto start = logits.data.begin()+b*p.c;
            float max = *std::max_element(start, start+p.c);
            double sum = 0;

            // softmax computation here
            for (std::size_t c = 0; c<p.c; ++c) {
                p(b, c) = std::exp(logits(b, c)-max);
                sum += p(b, c);
            }
            for (std::size_t c = 0; c<p.c; ++c) {
                p(b, c) = static_cast<float>(p(b, c)/sum);
            }
        }
        return p;
    }
    // build loss struct
    struct Loss {
        float value;
        Tensor gradient;
    };

    inline Loss cross_entropy(const Tensor& logits, const std::vector<std::size_t>& labels) {

        Tensor p = softmax(logits);
        double loss = 0;

        for (std::size_t b = 0; b<p.n; ++b) {

            float max = *std::max_element(logits.data.begin()+b*p.c, logits.data.begin()+(b+1)*p.c);
            double sum = 0;
            for (std::size_t c = 0; c<p.c; ++c) {
                sum += std::exp(static_cast<double>(logits(b, c)-max));
            }

            loss += std::log(sum)+static_cast<double>(max)-logits(b, labels[b]);
            p(b, labels[b]) -= 1;
        }
        for (auto& g :p.data) {
            g /= static_cast<float>(p.n);
        }
        return {
            static_cast<float>(loss/static_cast<double>(p.n)),
             std::move(p)
        };
    }

}
// namespace minitorch
