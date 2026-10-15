#pragma once
#include "../layer.hpp"

namespace minitorch {

    class MaxPool2D final:public Layer {
        std::size_t k_, stride_;
        Tensor input_shape_, output_shape_;
        std::vector<std::size_t> maxima_;
        public:
        explicit MaxPool2D(std::size_t kernel = 2, std::size_t stride = 2):k_(kernel), stride_(stride) {
            require(k_ && stride_, "Pool kernel/stride must be positive");
        }

        Tensor forward(const Tensor& x, bool training) override {
            x.validate();
            require(x.d == 1 && x.h >= k_ && x.w >= k_, "Pool kernel exceeds input");
            Tensor y(x.n, x.c, (x.h-k_)/stride_+1, (x.w-k_)/stride_+1);
            if (training) {
                input_shape_ = Tensor(x.n, x.c, x.h, x.w);
                output_shape_ = y;
                maxima_.resize(y.data.size());
            }
            for (std::size_t b = 0; b<x.n; ++b) for (std::size_t c = 0; c<x.c; ++c) for (std::size_t h = 0; h<y.h; ++h) for (std::size_t w = 0; w<y.w; ++w) {
                auto best = x.index(b, c, h*stride_, w*stride_);
                for (std::size_t ky = 0; ky<k_; ++ky) for (std::size_t kx = 0; kx<k_; ++kx) {
                    auto j = x.index(b, c, h*stride_+ky, w*stride_+kx);
                    if (x.data[j]>x.data[best]) best = j;
                }
                auto j = y.index(b, c, h, w);
                y.data[j] = x.data[best];
                if (training) maxima_[j] = best;
            }
            return y;
        }

        Tensor backward(const Tensor& dy) override {
            same_shape(dy, output_shape_);
            Tensor dx = input_shape_;
            for (std::size_t i = 0; i<dy.data.size(); ++i) dx.data[maxima_[i]] += dy.data[i];
            return dx;
        }

        void config(std::ostream& s) const override {
            s<<"MaxPool2D "<<k_<<' '<<stride_<<'\n';
        }
    };

}
// namespace minitorch
