#pragma once
#include "tensor.hpp"
#include <vector>

namespace minitorch {

    struct Dataset {
        Tensor images;
        std::vector<std::size_t> labels;
    };

    struct Metrics {
        float loss = 0, accuracy = 0;
    };

}
// namespace minitorch
