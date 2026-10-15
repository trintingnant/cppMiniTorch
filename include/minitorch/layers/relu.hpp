#pragma once
#include "../layer.hpp"
#include <algorithm>

namespace minitorch {

    class ReLU final:public Layer {
        Tensor input_;

        public:
            Tensor forward(const Tensor& x, bool training) override {
                x.validate();
                if (training) input_ = x;
                Tensor y = x;
                for (auto& v :y.data) v = std::max(0.f, v);
                return y;
            }

            Tensor backward(const Tensor& dy) override {
                same_shape(input_, dy);
                Tensor dx = dy;
                for (std::size_t i = 0; i<dx.data.size(); ++i) if (input_.data[i] <= 0) dx.data[i] = 0;
                return dx;
            }

            void config(std::ostream& s) const override {
                s<<"ReLU\n";
            }
    };

}
// namespace minitorch
