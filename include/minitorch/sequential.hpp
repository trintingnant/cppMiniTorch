#pragma once
#include "dataset.hpp"
#include "layer.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include "layers/dense.hpp"
#include "layers/relu.hpp"
#include "layers/flatten.hpp"
#include "layers/conv2d.hpp"
#include "layers/maxpool2d.hpp"
#include "layers/conv3d.hpp"
#include "layers/maxpool3d.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace minitorch {

    class Sequential {
        std::vector<std::unique_ptr<Layer>> layers_;
        public:
        template<class L, class... Args> Sequential& add(Args && ... args) {
            layers_.push_back(std::make_unique<L>(std::forward<Args>(args)...));
            return *this;
        }

        Tensor forward(Tensor x, bool training = false) {
            require(!layers_.empty(), "Model has no layers");
            x.validate();
            for (auto& l :layers_) x = l->forward(x, training);
            return x;
        }

        Tensor predict(const Tensor& x) {
            return softmax(forward(x));
        }

        std::vector<Parameter*> parameters() {
            std::vector<Parameter*> p;
            for (auto& l :layers_) {
                auto q = l->parameters();
                p.insert(p.end(), q.begin(), q.end());
            }
            return p;
        }

        float train_batch(const Tensor& x, const std::vector<std::size_t>& y, Optimizer& optimizer) {
            auto loss = cross_entropy(forward(x, true), y);
            Tensor grad = std::move(loss.gradient);
            for (auto l = layers_.rbegin(); l != layers_.rend(); ++l) grad = (*l)->backward(grad);
            optimizer.step(parameters());
            return loss.value;
        }

        static Dataset batch(const Dataset& d, const std::vector<std::size_t>& indices, std::size_t start, std::size_t count) {
            d.images.validate();
            require(d.labels.size() == d.images.n && count && start <= indices.size() && count <= indices.size()-start, "Invalid batch");
            Dataset b {
                Tensor(count, d.images.c, d.images.d, d.images.h, d.images.w), {}
            };
            b.labels.reserve(count);
            auto size = d.images.sample_size();
            for (std::size_t i = 0; i<count; ++i) {
                auto j = indices[start+i];
                require(j<d.images.n, "Invalid sample index");
                std::copy_n(d.images.data.begin()+j*size, size, b.images.data.begin()+i*size);
                b.labels.push_back(d.labels[j]);
            }
            return b;
        }

        Metrics evaluate(const Dataset& d, std::size_t batch_size = 64) {
            require(batch_size>0, "Batch size must be positive");
            d.images.validate();
            require(d.images.n>0 && d.labels.size() == d.images.n, "Evaluation requires a nonempty, labeled dataset");
            std::vector<std::size_t> ids(d.images.n);
            std::iota(ids.begin(), ids.end(), 0);
            double loss = 0;
            std::size_t correct = 0;
            for (std::size_t i = 0; i<ids.size(); i += batch_size) {
                auto b = batch(d, ids, i, std::min(batch_size, ids.size()-i));
                auto z = forward(b.images);
                loss += static_cast<double>(cross_entropy(z, b.labels).value)*static_cast<double>(b.images.n);
                for (std::size_t j = 0; j<z.n; ++j) {
                    auto begin = z.data.begin()+j*z.c;
                    correct += static_cast<std::size_t>(std::max_element(begin, begin+z.c)-begin) == b.labels[j];
                }
            }
            return {static_cast<float>(loss/static_cast<double>(ids.size())), static_cast<float>(static_cast<double>(correct)/static_cast<double>(ids.size()))};
        }

        void fit(const Dataset& d, Optimizer& optimizer, std::size_t epochs = 5, std::size_t batch_size = 32, unsigned seed = 42, const std::function<void(std::size_t, float)>& on_epoch = {}) {
            d.images.validate();
            require(d.images.n>0 && d.labels.size() == d.images.n && batch_size>0, "Invalid or empty dataset/batch size");
            std::vector<std::size_t> ids(d.images.n);
            std::iota(ids.begin(), ids.end(), 0);
            std::mt19937 rng(seed);
            for (std::size_t e = 0; e<epochs; ++e) {
                std::shuffle(ids.begin(), ids.end(), rng);
                double loss = 0;
                for (std::size_t i = 0; i<ids.size(); i += batch_size) {
                    auto b = batch(d, ids, i, std::min(batch_size, ids.size()-i));
                    loss += static_cast<double>(train_batch(b.images, b.labels, optimizer))*static_cast<double>(b.images.n);
                }
                if (on_epoch) on_epoch(e+1, static_cast<float>(loss/static_cast<double>(ids.size())));
            }
        }

        void save(const std::string& path) {
            require(!layers_.empty(), "Cannot save an empty model");
            std::ofstream s(path);
            require(bool(s), "Cannot open model for writing: "+path);
            s<<"MINITORCH 1\n"<<layers_.size()<<'\n'<<std::setprecision(std::numeric_limits<float>::max_digits10);
            for (auto& l :layers_) {
                l->config(s);
                for (auto p:l->parameters()) {
                    s<<p->value.size()<<'\n';
                    for (float v:p->value) {
                        require(std::isfinite(v), "Cannot save non-finite parameter");
                        s<<v<<' ';
                    }
                    s<<'\n';
                }
            }
            s.flush();
            require(bool(s), "Model write failed");
        }

        static Sequential load(const std::string& path) {
            std::ifstream s(path);
            require(bool(s), "Cannot open model: "+path);
            std::string magic;
            std::size_t version = 0, count = 0;
            s>>magic>>version>>count;
            require(s && (magic == "MINITORCH" || magic == "MINIKERAS") && version == 1 && count>0 && count <= 1024, "Invalid model header");
            Sequential model;
            auto dim = [&]() {
                std::size_t v = 0;
                s>>v;
                require(s && v>0 && v <= 1000000, "Invalid model dimension");
                return v;
            };
            std::size_t total = 0;
            auto reserve_parameters = [&](std::size_t weights, std::size_t biases) {
                require(weights <= 10000000 && biases <= 10000000-weights && total <= 10000000-weights-biases, "Model parameter limit exceeded");
                total += weights+biases;
            };
            for (std::size_t j = 0; j<count; ++j) {
                std::string type;
                s>>type;
                std::unique_ptr<Layer> layer;
                if (type == "Dense") {
                    auto in = dim(), out = dim();
                    reserve_parameters(volume({in, out}), out);
                    layer = std::make_unique<Dense>(in, out);
                }
                else if (type == "Conv2D") {
                    auto in = dim(), out = dim(), k = dim(), stride = dim();
                    std::size_t pad = 0;
                    s>>pad;
                    require(s && pad <= 1000000, "Invalid model padding");
                    reserve_parameters(volume({in, out, k, k}), out);
                    layer = std::make_unique<Conv2D>(in, out, k, stride, pad);
                }
                else if (type == "Conv3D") {
                    auto in = dim(), out = dim(), k = dim(), stride = dim();
                    std::size_t pad = 0;
                    s>>pad;
                    require(s && pad <= 1000000, "Invalid model padding");
                    reserve_parameters(volume({in, out, k, k, k}), out);
                    layer = std::make_unique<Conv3D>(in, out, k, stride, pad);
                }
                else if (type == "MaxPool3D") {
                    auto k = dim(), stride = dim();
                    layer = std::make_unique<MaxPool3D>(k, stride);
                }
                else if (type == "ReLU") layer = std::make_unique<ReLU>();
                else if (type == "Flatten") layer = std::make_unique<Flatten>();
                else if (type == "MaxPool2D") {
                    auto k = dim(), stride = dim();
                    layer = std::make_unique<MaxPool2D>(k, stride);
                }
                else throw std::runtime_error("Unknown model layer: "+type);
                for (auto p:layer->parameters()) {
                    std::size_t size = 0;
                    s>>size;
                    require(s && size == p->value.size(), "Invalid parameter count");
                    for (auto& v :p->value) {
                        s>>v;
                        require(s && std::isfinite(v), "Invalid/truncated model parameters");
                    }
                }
                model.layers_.push_back(std::move(layer));
            }
            s>>std::ws;
            require(s.eof(), "Trailing model data");
            return model;
        }
    };

}
// namespace minitorch
