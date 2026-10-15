/* 
Standard 2d convolutional layers for image data
*/

#pragma once
#include "../layer.hpp"
#include <limits>

namespace minitorch {

    class Conv2D final:public Layer {
        std::size_t in_, out_, k_, stride_, pad_;
        Tensor input_;

        // compute output size with stride + pad 
        std::size_t output_size(std::size_t d) const {
            require(pad_ <= (std::numeric_limits<std::size_t>::max()-d)/2, 
                            "padding overflow"
                            );

            require(d+2*pad_ >= k_, 
                    "kernel exceeds padded input"
                    );
            return (d+2*pad_-k_)/stride_+1;
        }

        public:
            Parameter weights, bias;
            // constructor
            Conv2D(std::size_t in, std::size_t out, std::size_t kernel, std::size_t stride = 1, std::size_t padding = 0, unsigned seed = 42)
            :in_(in), out_(out), k_(kernel), stride_(stride), pad_(padding), weights(volume({out, in, kernel, kernel})), bias(out) {
                weights.initialize(volume({in, kernel, kernel}), seed);
            }

        Tensor forward(const Tensor& x, bool training) override {
            x.validate();
            require(x.c == in_ && 
                    x.d == 1, 
                    "channels don't match"
                    );
            // this is naive convolution and will almost certainly run to large runtimes
           Tensor y(x.n, out_, output_size(x.h), output_size(x.w));
                
                /*
                during training we maintain a copy of the input in the layer
                for backpropagation 
                */
                if (training) {
                    input_ = x;
                }
                // this is qite ugly (and painful to read, obviously)
                for (std::size_t b = 0; b < x.n; ++b) {
                    for (std::size_t o = 0; o < out_; ++o) {
                        for (std::size_t y0 = 0; y0 < y.h; ++y0) {
                            for (std::size_t x0 = 0; x0 < y.w; ++x0) {

                                float sum = bias.value[o];

                                for (std::size_t i = 0; i < in_; ++i) {
                                    for (std::size_t ky = 0; ky < k_; ++ky) {
                                        for (std::size_t kx = 0; kx < k_; ++kx) {

                                            auto iy = y0 * stride_ + ky;
                                            auto ix = x0 * stride_ + kx;

                                            if (
                                                iy >= pad_ &&
                                                ix >= pad_ &&
                                                iy - pad_ < x.h &&
                                                ix - pad_ < x.w
                                            ) {
                                                std::size_t weight_idx =
                                                    ((o * in_ + i) * k_ + ky) * k_ + kx;

                                                sum += x(b, i, iy - pad_, ix - pad_) *
                                                    weights.value[weight_idx];
                                            }
                                        }
                                    }
                                }

                                y(b, o, y0, x0) = sum;
                            }
                        }
                    }
                }
        return y;
        }


        Tensor backward(const Tensor& dy) override {
            dy.validate();
            require(
                dy.n == input_.n && 
                dy.c == out_ && 
                dy.d == 1 && 
                dy.h == output_size(input_.h) && 
                dy.w == output_size(input_.w), 
                "shape mismatch for backward pass"
                );
                
            // derivatives
            Tensor dx(input_.n, in_, input_.h, input_.w);
            weights.zero_grad();
            bias.zero_grad();
            for (std::size_t b = 0; b<dy.n; ++b) for (std::size_t o = 0; o<out_; ++o)
            for (std::size_t y = 0; y<dy.h; ++y) for (std::size_t x = 0; x<dy.w; ++x) {
                float g = dy(b, o, y, x);
                bias.grad[o] += g;
                for (std::size_t i = 0; i<in_; ++i) for (std::size_t ky = 0; ky<k_; ++ky) for (std::size_t kx = 0; kx<k_; ++kx) {
                    auto iy = y*stride_+ky, ix = x*stride_+kx;
                    if (iy >= pad_ && ix >= pad_ && iy-pad_<input_.h && ix-pad_<input_.w) {
                        auto j = ((o*in_+i)*k_+ky)*k_+kx;
                        weights.grad[j] += input_(b, i, iy-pad_, ix-pad_)*g;
                        dx(b, i, iy-pad_, ix-pad_) += weights.value[j]*g;
                    }
                }
            }
            return dx;
        }

        std::vector<Parameter*> parameters() override {
            return {&weights, &bias};
        }

        void config(std::ostream& s) const override {
            s<<"Conv2D "<<in_<<' '<<out_<<' '<<k_<<' '<<stride_<<' '<<pad_<<'\n';
        }
    };

}
// namespace minitorch
