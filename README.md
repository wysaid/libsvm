# LibSVM - A Library for Support Vector Machines

[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](COPYRIGHT)

LibSVM is a simple, easy-to-use, and efficient software for SVM classification and regression. It solves C-SVM classification, nu-SVM classification, one-class-SVM, epsilon-SVM regression, and nu-SVM regression. It also provides an automatic model selection tool for C-SVM classification.

**Homepage**: https://www.csie.ntu.edu.tw/~cjlin/libsvm

Please read the [COPYRIGHT](COPYRIGHT) file before using LibSVM.

## About This Fork

This is a modernized fork of LibSVM with the following improvements:

- **CMake Build System**: Replaced traditional Makefiles with modern CMake (3.16+) for better cross-platform support
- **Reorganized Structure**: Cleaner directory layout with `src/`, `apps/`, `bindings/`, and `examples/`
- **Unified Language Bindings**: CMake-based build configuration for Python, Java, and MATLAB bindings
- **Removed Precompiled Binaries**: All binaries are now built from source for better security and compatibility
- **Modern Documentation**: Updated to Markdown format with comprehensive build instructions

For the original build instructions using Makefile, see [README.original](README.original).

For development logs and migration details, see [docs/DEV_LOG.md](docs/DEV_LOG.md).

## Table of Contents

- [Quick Start](#quick-start)
- [Building with CMake](#building-with-cmake)
- [Installation](#installation)
- [Data Format](#data-format)
- [Command-Line Tools](#command-line-tools)
- [Library Usage](#library-usage)
- [Language Bindings](#language-bindings)
- [Examples](#examples)
- [Citation](#citation)

## Quick Start

If you are new to SVM and if the data is not large, please use `easy.py` in the `tools` directory after installation. It does everything automatically -- from data scaling to parameter selection.

```bash
python tools/easy.py training_file [testing_file]
```

More information about parameter selection can be found in [tools/README](tools/README).

## Building with CMake

LibSVM uses CMake as its build system. Minimum required CMake version is 3.16.

### Basic Build

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_SHARED_LIBS` | ON | Build shared library instead of static |
| `LIBSVM_ENABLE_OPENMP` | OFF | Enable OpenMP for parallel training |
| `LIBSVM_BUILD_APPS` | ON | Build command-line tools |
| `LIBSVM_BUILD_EXAMPLES` | OFF | Build example programs (svm-toy) |
| `LIBSVM_BUILD_PYTHON` | OFF | Build Python bindings |
| `LIBSVM_BUILD_JAVA` | OFF | Build Java bindings |
| `LIBSVM_BUILD_MATLAB` | OFF | Build MATLAB/MEX bindings |

### Build with Options

```bash
# Build with OpenMP support and examples
cmake -DLIBSVM_ENABLE_OPENMP=ON -DLIBSVM_BUILD_EXAMPLES=ON ..
cmake --build .

# Build all language bindings
cmake -DLIBSVM_BUILD_PYTHON=ON -DLIBSVM_BUILD_JAVA=ON ..
cmake --build .

# Build MATLAB bindings (requires MATLAB)
cmake -DLIBSVM_BUILD_MATLAB=ON -DMatlab_ROOT_DIR=/path/to/matlab ..
cmake --build . --target matlab-bindings
```

### Build svm-toy (Qt GUI)

The svm-toy example requires Qt5 or Qt6:

```bash
cmake -DLIBSVM_BUILD_EXAMPLES=ON ..
cmake --build . --target svm-toy
```

## Installation

```bash
cmake --install . --prefix /usr/local
```

This installs:
- `libsvm` library to `lib/`
- `svm.h` header to `include/`
- Command-line tools to `bin/`
- CMake package config files for `find_package(LibSVM)`

### Using LibSVM in Your CMake Project

```cmake
find_package(LibSVM REQUIRED)
target_link_libraries(your_target LibSVM::svm)
```

## Data Format

The format of training and testing data files is:

```
<label> <index1>:<value1> <index2>:<value2> ...
```

- **Classification**: `<label>` is an integer indicating the class label
- **Regression**: `<label>` is the target value (any real number)
- **One-class SVM**: `<label>` has no effect

The pair `<index>:<value>` gives a feature value. Indices must be in **ascending order** and start from 1.

Example data file is provided at `examples/data/heart_scale`.

## Command-Line Tools

After building, the following tools are available in `build/bin/`:

### svm-train

```bash
svm-train [options] training_set_file [model_file]
```

Options:
- `-s svm_type`: SVM type (0: C-SVC, 1: nu-SVC, 2: one-class, 3: epsilon-SVR, 4: nu-SVR)
- `-t kernel_type`: Kernel type (0: linear, 1: polynomial, 2: RBF, 3: sigmoid, 4: precomputed)
- `-c cost`: Set parameter C (default 1)
- `-g gamma`: Set gamma in kernel function (default 1/num_features)
- `-v n`: n-fold cross validation mode
- `-q`: Quiet mode

### svm-predict

```bash
svm-predict [options] test_file model_file output_file
```

### svm-scale

```bash
svm-scale [options] data_filename
```

Options:
- `-l lower`: x scaling lower limit (default -1)
- `-u upper`: x scaling upper limit (default +1)
- `-s save_filename`: Save scaling parameters
- `-r restore_filename`: Restore scaling parameters

## Library Usage

Include `svm.h` in your C/C++ source files and link with `libsvm`:

```c
#include "svm.h"

// Train a model
struct svm_model *model = svm_train(&prob, &param);

// Make predictions
double result = svm_predict(model, x);

// Save/load model
svm_save_model("model.txt", model);
struct svm_model *loaded = svm_load_model("model.txt");

// Cleanup
svm_free_and_destroy_model(&model);
```

See `apps/svm-train.c` and `apps/svm-predict.c` for complete examples.

## Language Bindings

### Python

The Python bindings use ctypes to load the libsvm shared library.

```bash
# Build the shared library
cmake -DBUILD_SHARED_LIBS=ON -DLIBSVM_BUILD_PYTHON=ON ..
cmake --build .

# Use the bindings
cd bindings/python
export LD_LIBRARY_PATH=/path/to/build/lib:$LD_LIBRARY_PATH  # Linux
export DYLD_LIBRARY_PATH=/path/to/build/lib:$DYLD_LIBRARY_PATH  # macOS
python -c "from libsvm import svmutil; svmutil.svm_train(...)"
```

See [bindings/python/README](bindings/python/README) for details.

### Java

The Java version is a pure Java implementation (does not require the C library).

```bash
cmake -DLIBSVM_BUILD_JAVA=ON ..
cmake --build . --target java-bindings

# Run
java -classpath build/bindings/java/libsvm.jar svm_train [arguments]
java -classpath build/bindings/java/libsvm.jar svm_predict [arguments]
```

### MATLAB/Octave

```bash
cmake -DLIBSVM_BUILD_MATLAB=ON -DMatlab_ROOT_DIR=/path/to/matlab ..
cmake --build . --target matlab-bindings
```

Or use the `make.m` script in MATLAB/Octave. See [bindings/matlab/README](bindings/matlab/README) for details.

## Examples

### Training and Prediction

```bash
# Scale data
./build/bin/svm-scale -l -1 -u 1 -s range examples/data/heart_scale > heart_scale.scaled

# Train
./build/bin/svm-train heart_scale.scaled

# Predict
./build/bin/svm-predict heart_scale.scaled heart_scale.scaled.model output
```

### Cross Validation

```bash
./build/bin/svm-train -v 5 examples/data/heart_scale
```

### Different SVM Types

```bash
# C-SVC with RBF kernel
./build/bin/svm-train -s 0 -t 2 -c 100 -g 0.1 data_file

# epsilon-SVR with linear kernel
./build/bin/svm-train -s 3 -t 0 -p 0.1 data_file
```

## Project Structure

```
libsvm/
├── CMakeLists.txt          # Main CMake configuration
├── src/                    # Core library source
│   ├── svm.cpp
│   └── svm.h
├── apps/                   # Command-line tools
│   ├── svm-train.c
│   ├── svm-predict.c
│   └── svm-scale.c
├── examples/               # Example programs
│   ├── data/heart_scale    # Sample dataset
│   └── svm-toy/            # Qt GUI demo
├── bindings/               # Language bindings
│   ├── python/
│   ├── java/
│   └── matlab/
├── tools/                  # Python utility scripts
└── docs/                   # Documentation
```

## Citation

If you find LibSVM helpful, please cite it as:

> Chih-Chung Chang and Chih-Jen Lin, LIBSVM: a library for support vector machines. ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011.

Software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm

## License

LibSVM is distributed under the BSD 3-Clause License. See [COPYRIGHT](COPYRIGHT) for details.

## Contact

For questions and comments, please email cjlin@csie.ntu.edu.tw
