#pragma once
#include "parameter.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace minitorch {

    class Optimizer {
        public:
        virtual ~Optimizer() = default;
        virtual void step(const std::vector<Parameter*>&) = 0;
    };
    // SGD with optional momentum and Nesterov acceleration.

    class SGD final:public Optimizer {
        float lr_;
        float momentum_;
        bool nesterov_;


    public:
        explicit SGD(
            float lr = .01f,
            float momentum = 0.0f,
            bool nesterov = false
        )
            : lr_(lr),
            momentum_(momentum),
            nesterov_(nesterov) {

                require(
                    lr > 0 && std::isfinite(lr),
                    "Learning rate must be real and finite"
                );

                require(
                    momentum >= 0 && momentum < 1,
                    "Momentum must be between 0 and 1 (exclusive)"
                );

                require(
                    !nesterov || momentum > 0,
                    "Need positive momentum for Nesterov"
                );

            }


        void step(const std::vector<Parameter*>& ps) override {
            for (auto p : ps) {
                require(
                    p->grad.size() == p->value.size() &&
                    p->m.size() == p->value.size(),
                    "Parameter sizes don't match"
                );

                for (std::size_t i = 0; i<p->value.size(); ++i) {
                    
                    //gradient
                    float g =  p->grad[i];

                    //vanilla SGD
                    if (momentum_ == 0.0f){
                        p->value[i] -= lr_ * g;
                        continue;
                    }

                    //momentum 
                    p->m[i] = momentum_ * p->m[i] + g;

                    if (nesterov_){
                        g += momentum_ * p->m[i];
                    } else {
                        g = p->m[i];

                    }

                    p->value[i] -= lr_ * g; 
                }
            }
        }
    };
    
    class Adam final:public Optimizer {
        float lr_, b1_, b2_, epsilon_;
        std::size_t t_ = 0;

        public:
            explicit Adam(
                float lr = .001f, 
                float b1 = .9f, 
                float b2 = .999f, 
                float epsilon = 1e-8f
            )
                : lr_(lr), 
                b1_(b1), 
                b2_(b2), 
                epsilon_(epsilon) {

                    require(lr>0 && std::isfinite(lr) && 
                            b1 >= 0 && b1<1 && 
                            b2 >= 0 && b2<1 && 
                            epsilon>0 && 
                            std::isfinite(epsilon), 
                            "Invalid Adam configuration"
                        );
                }

        void step(const std::vector<Parameter*>& ps) override {

            if (t_ == 0) {
                for (auto p : ps) {
                    std::fill(p->m.begin(), p->m.end(), 0);
                    std::fill(p->v.begin(), p->v.end(), 0);
                }
            }
            ++t_;
            double a = 1-std::pow(static_cast<double>(b1_), t_); 
            double b = 1-std::pow(static_cast<double>(b2_), t_);
            for (auto p : ps) for (std::size_t i = 0; i<p->value.size(); ++i) {
                float g = p->grad[i];
                p->m[i] = b1_*p->m[i]+(1-b1_)*g;
                p->v[i] = b2_*p->v[i]+(1-b2_)*g*g;
                p->value[i] -= static_cast<float>(lr_*(p->m[i]/a)/(std::sqrt(p->v[i]/b)+epsilon_));
            }
        }
    
    };


}
// namespace minitorch
