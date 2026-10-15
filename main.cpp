// Build: clang++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Iinclude main.cpp -o mnist
// Run:   ./mnist [data_directory]
#include "minitorch.hpp"
#include <algorithm>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    using namespace minitorch;
    try {
        if (argc > 2) {
            std::cerr << "Usage: " << argv[0] << " [data_directory]\n";
            return 1;
        }
        const std::string directory = argc == 2 ? argv[1] : "data";

        // we'll train only on small subsets.

        constexpr std::size_t train_limit = 1000;
        constexpr std::size_t test_limit = 1000;
        constexpr std::size_t epochs = 5;
        constexpr std::size_t batch_size = 32;

        std::cout << "Loading MNIST from " << directory << "..." << std::endl;

        // test/train load 
        const Dataset train = load_idx(directory + "/train-images-idx3-ubyte",
                                       directory + "/train-labels-idx1-ubyte", train_limit);
        const Dataset test = load_idx(directory + "/t10k-images-idx3-ubyte",
                                      directory + "/t10k-labels-idx1-ubyte", test_limit);

        std::cout << "Training images: " << train.images.n
                  << " | Test images: " << test.images.n << '\n';

        // Normalized grayscale images -> 784 pixels -> 128 hidden units -> 10 logits.

        // setup
        Sequential model;
        model.add<Flatten>()
             .add<Dense>(28 * 28, 128, 42)
             .add<ReLU>()
             .add<Dense>(128, 10, 43);
        Adam optimizer(0.001f);
        
        model.fit(train, optimizer, epochs, batch_size, 42,
                  [](std::size_t epoch, float loss) {
                      std::cout << "Epoch " << epoch << " | loss: " << loss << std::endl;
                  });

        const Metrics result = model.evaluate(test, batch_size);
        std::cout << "Test loss: " << result.loss
                  << "\nTest accuracy: " << result.accuracy * 100.0f << "%\n";

        model.save("mnist_model.txt");
        std::cout << "Saved model to mnist_model.txt\n";

    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
