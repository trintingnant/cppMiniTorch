#pragma once
#include "../dataset.hpp"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <istream>
#include <string>
#include <vector>

/* 

IDX-file reader for dataset loading 
    read_u23: just reads a 32-bit unsigned integer
    load_idx: loads datasets such as MNIST/FashionMNIST
        -> converts imagines to (normalised) float vectors
*/

namespace minitorch {

    inline std::uint32_t read_u32(std::istream& s) {
        unsigned char b[4] = {};
        s.read(reinterpret_cast<char*>(b), 4);
        require(bool(s), "Truncated IDX header");
        return (std::uint32_t(b[0])<<24)|(std::uint32_t(b[1])<<16)|(std::uint32_t(b[2])<<8)|b[3];
    }

    inline Dataset load_idx(const std::string& images, const std::string& labels, std::size_t limit = 0) {
        std::ifstream x(images, std::ios::binary), y(labels, std::ios::binary);
        require(read_u32(x) == 2051 && read_u32(y) == 2049, "Invalid IDX magic numbers");
        const auto n = read_u32(x), m = read_u32(y);
        require(n > 0 && n == m, "Invalid IDX sample counts");
        const auto h = read_u32(x), w = read_u32(x);
        std::size_t count = limit ? std::min<std::size_t>(limit, n) : n;

        Dataset d {
            Tensor(count, 1, h, w), std::vector<std::size_t>(count)
        };

        std::vector<unsigned char> row(volume({h, w}));

        for (std::size_t i = 0; i<count; ++i) {
            x.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row.size()));
            unsigned char label = 0;
            y.read(reinterpret_cast<char*>(&label), 1);
            require(bool(x) && bool(y), "Failed to read IDX sample");
            d.labels[i] = label;
            for (std::size_t j = 0; j<row.size(); ++j) d.images.data[i*row.size()+j] = row[j]/255.f;
        }
        return d;
    }

}
// namespace minitorch
