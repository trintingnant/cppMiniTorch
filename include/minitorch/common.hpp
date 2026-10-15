#pragma once
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace minitorch {

    inline void require(bool ok, const std::string& message) {
        if (!ok) throw std::runtime_error(message);
    }


    inline std::size_t volume(std::initializer_list<std::size_t> dims) {
        std::size_t n = 1;
        for (auto d:dims) {
            require(d && n <= std::numeric_limits<std::size_t>::max()/d, "Invalid/overflowing shape");
            n *= d;
        }
        return n;
    }

}
// namespace minitorch

