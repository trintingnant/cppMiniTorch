# cppMiniTorch

Pytorch-style, fully C++17 native. 

Disclaimer: this is an educational, bare-bones 'PyTorch-style' project. 
It **isn't** PyTorch. 

**Scope:** No 3D convolution/pooling, no GPU backend, no SIMD, no arbitrary computational graphs, no automatic differentiation engine, etc. In short: it has none of the features that make PyTorch an okay-ish ML platform. Hell, it doesn't even have: Dropout, regularisers, 3D-convolutions, or anything over and above what's needed to learn more complex stuff than MNIST/FashionMNIST. 


## Quick start: 

### 1. Download and extract the four IDX files

Run from the **project root**:

```bash
mkdir -p data/fashion-mnist
cd data/fashion-mnist

BASE_URL="https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"

for file in \
    train-images-idx3-ubyte.gz \
    train-labels-idx1-ubyte.gz \
    t10k-images-idx3-ubyte.gz \
    t10k-labels-idx1-ubyte.gz
do
    curl -fL --retry 3 -o "$file" "$BASE_URL/$file" || exit 1
done

gunzip -f ./*.gz
cd ../..
```

Check that the following files exist:

```text
data/fashion-mnist/train-images-idx3-ubyte
data/fashion-mnist/train-labels-idx1-ubyte
data/fashion-mnist/t10k-images-idx3-ubyte
data/fashion-mnist/t10k-labels-idx1-ubyte
```

### 2. Point `main.cpp` at Fashion-MNIST

Replace the two existing `load_idx()` calls with:

```cpp
Dataset train = load_idx(
    "data/fashion-mnist/train-images-idx3-ubyte",
    "data/fashion-mnist/train-labels-idx1-ubyte"
);

Dataset test = load_idx(
    "data/fashion-mnist/t10k-images-idx3-ubyte",
    "data/fashion-mnist/t10k-labels-idx1-ubyte"
);
```

### 3. Compile with Clang++ 

```bash
clang++ -std=c++17 -O3 -Wall -Wextra -Wpedantic \
    -Iinclude main.cpp -o fashion_mnist
```

### 4. Or compile with GCC++

```bash
g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic \
    -Iinclude main.cpp -o fashion_mnist
```

On macOS, `g++` might be an alias for Apple Clang. Use a Homebrew-installed versioned GCC executable (such as `g++-15`) if you specifically want GCC.

## Example: models can be built in PyTorch-style, e.g.

```cpp
    Sequential model;

    model.add<Conv2D>(1, 16, 3)
         .add<ReLU>()
         .add<MaxPool2D>(2)
         .add<Conv2D>(16, 32, 3)
         .add<ReLU>()
         .add<MaxPool2D>(2)
         .add<Flatten>()
         .add<Dense>(800, 128)
         .add<ReLU>()
         .add<Dense>(128, 10);


```

