#pragma once
#include "../layer.hpp"

namespace minitorch {

    class Dense final:public Layer {
        std::size_t in_, out_;
        Tensor input_;
        public:
        Parameter weights, bias;
        Dense(std::size_t in, std::size_t out, unsigned seed = 42):in_(in), out_(out), weights(volume({in, out})), bias(out) {
            weights.initialize(in, seed);
        }

        Tensor forward(const Tensor& x, bool training) override {
            x.validate();
            require(
                x.c == in_ && 
                x.d == 1 && 
                x.h == 1 && 
                x.w == 1, 
                "flattened input features shape mismatch"
                );
            // same as elsewhere: maintain copy

            if (training) {input_ = x;}
            // build output tensor 
            Tensor y(x.n, out_);
            for (std::size_t b = 0; b<x.n; ++b) for (std::size_t o = 0; o<out_; ++o) {
                float s = bias.value[o];
                for (std::size_t i = 0; i<in_; ++i) s += x(b, i)*weights.value[o*in_+i];
                    y(b, o) = s;
                }
                return y;
            }
        // backward pass
        Tensor backward(const Tensor& dy) override {
            require(
                input_.n>0, 
                "need a forward pass"
                );
            dy.validate();
            require(
                dy.n == input_.n && 
                dy.c == out_ && 
                dy.d == 1 && 
                dy.h == 1 && 
                dy.w == 1,
                "dense shape mismatch"
            );
            Tensor dx(input_.n, in_);
            weights.zero_grad();
            bias.zero_grad();
            for (std::size_t b = 0; b<dy.n; ++b) {
                for (std::size_t o = 0; o<out_; ++o) {
                    float g = dy(b, o);
                    bias.grad[o] += g;
                    for (std::size_t i = 0; i<in_; ++i) {
                        weights.grad[o*in_+i] += g*input_(b, i);
                        dx(b, i) += g*weights.value[o*in_+i];
                    }
                }
            }
            return dx;
            }
        

        std::vector<Parameter*> parameters() override {
            return {&weights, &bias};
        }

        void config(std::ostream& s) const override {
            s<<"Dense "<<in_<<' '<<out_<<'\n';
        }
    };

}
// namespace minitorch
