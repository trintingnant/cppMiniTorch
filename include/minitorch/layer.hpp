#pragma once
#include "tensor.hpp"
#include "parameter.hpp"
#include <ostream>
#include <vector>

namespace minitorch {

    class Layer {
        public:
        virtual ~Layer() = default;
        virtual Tensor forward(const Tensor&, bool training) = 0;
        virtual Tensor backward(const Tensor&) = 0;
        virtual std::vector<Parameter*> parameters() {
            return {};
        }

        virtual void config(std::ostream&) const = 0;
    };

}
// namespace minitorch
