#pragma once
#include "../layer.hpp"

namespace minitorch {

    class Flatten final:public Layer {
        std::size_t n_ = 0, c_ = 0, d_ = 0, h_ = 0, w_ = 0;
        public:
        Tensor forward(const Tensor& x, bool training) override {
            x.validate();
            if (training) {
                n_ = x.n;
                c_ = x.c;
                d_ = x.d;
                h_ = x.h;
                w_ = x.w;
            }

            Tensor y = x;
            y.c = x.sample_size();
            y.d = y.h = y.w = 1;
            return y;
        }

        Tensor backward(const Tensor& dy) override {
            dy.validate();
            require(n_ && dy.n == n_ && dy.c == volume({c_, d_, h_, w_}) && dy.d == 1 && dy.h == 1 && dy.w == 1, "Flatten backward shape mismatch");
            Tensor dx = dy;
            dx.c = c_;
            dx.d = d_;
            dx.h = h_;
            dx.w = w_;
            return dx;
        }

        void config(std::ostream& s) const override {
            s<<"Flatten\n";
        }
    };

}
// namespace minitorch
