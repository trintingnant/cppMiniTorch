/* basic tensor definitions 
uses a torch-style tensor convention:
    n: batch
    c: channel
    d: depth
    h: height
    w: width 
*/

#pragma once
#include "common.hpp"
#include <vector>

namespace minitorch {

    struct Tensor {
        // 2D-instances have d=1. 
        std::size_t n = 0, c = 0, d = 1, h = 0, w = 0;
        std::vector<float> data;
        Tensor() = default;
        Tensor(std::size_t n_, std::size_t c_, std::size_t h_ = 1, std::size_t w_ = 1)
        :Tensor(n_, c_, 1, h_, w_) {
        }

        Tensor(std::size_t n_, std::size_t c_, std::size_t d_, std::size_t h_, std::size_t w_)
        :n(n_), c(c_), d(d_), h(h_), w(w_), data(volume({n, c, d, h, w}), 0) {
        }

        std::size_t sample_size() const {
            return volume({c, d, h, w});
        }

        void validate() const {
            require(data.size() == volume({n, c, d, h, w}), 
                    "Tensor storage does not match shape"
                    );
        }

        std::size_t index(std::size_t a, std::size_t b, std::size_t y, std::size_t x) const {
            require(d == 1, "need 5 indices");
            return index(a, b, 0, y, x);
        }
        std::size_t index(std::size_t a, std::size_t b, std::size_t z, std::size_t y, std::size_t x) const {
            require(a<n && b<c && z<d && y<h && x<w, "Tensor index out of bounds");
            return (((a*c+b)*d+z)*h+y)*w+x;
        }
        // operators for 5d- and 4d-Tensors
        float& operator()(std::size_t a, std::size_t b, std::size_t z, std::size_t y, std::size_t x) {
            return data[index(a, b, z, y, x)];
        }

        float operator()(std::size_t a, std::size_t b, std::size_t z, std::size_t y, std::size_t x) const {
            return data[index(a, b, z, y, x)];
        }

        float& operator()(std::size_t a, std::size_t b, std::size_t y = 0, std::size_t x = 0) {
            return data[index(a, b, y, x)];
        }

        float operator()(std::size_t a, std::size_t b, std::size_t y = 0, std::size_t x = 0) const {
            return data[index(a, b, y, x)];
        }
    };

    inline void same_shape(const Tensor& a, const Tensor& b) {
        a.validate();
        b.validate();
        require(a.n == b.n && 
                a.c == b.c && 
                a.d == b.d && 
                a.h == b.h && 
                a.w == b.w, 
                "Tensor shape mismatch"
            );
    }

}
// namespace minitorch
