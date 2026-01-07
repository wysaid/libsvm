/**
 * @file test_utils.cpp
 * @brief Implementation of common test utilities
 */

#include "test_utils.h"
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <filesystem>

namespace libsvm_test {

// ============================================================================
// SvmProblemBuilder Implementation
// ============================================================================

SvmProblemBuilder::~SvmProblemBuilder() {
    clear();
}

void SvmProblemBuilder::addSample(double label, const std::vector<std::pair<int, double>>& features) {
    labels_.push_back(label);
    
    std::vector<svm_node> nodes;
    nodes.reserve(features.size() + 1);
    
    for (const auto& [idx, val] : features) {
        svm_node node;
        node.index = idx;
        node.value = val;
        nodes.push_back(node);
    }
    
    // Terminating node
    svm_node end_node;
    end_node.index = -1;
    end_node.value = 0;
    nodes.push_back(end_node);
    
    node_vectors_.push_back(std::move(nodes));
    built_ = false;
}

void SvmProblemBuilder::addDenseSample(double label, const std::vector<double>& features) {
    std::vector<std::pair<int, double>> sparse;
    sparse.reserve(features.size());
    
    for (size_t i = 0; i < features.size(); ++i) {
        if (features[i] != 0.0) {  // Skip zeros for sparse representation
            sparse.emplace_back(static_cast<int>(i + 1), features[i]);
        }
    }
    
    addSample(label, sparse);
}

svm_problem* SvmProblemBuilder::build() {
    if (labels_.empty()) {
        return nullptr;
    }
    
    // Build x pointers
    x_ptrs_.clear();
    x_ptrs_.reserve(node_vectors_.size());
    for (auto& nodes : node_vectors_) {
        x_ptrs_.push_back(nodes.data());
    }
    
    problem_.l = static_cast<int>(labels_.size());
    problem_.y = labels_.data();
    problem_.x = x_ptrs_.data();
    
    built_ = true;
    return &problem_;
}

void SvmProblemBuilder::clear() {
    labels_.clear();
    node_vectors_.clear();
    x_ptrs_.clear();
    problem_ = svm_problem{};
    built_ = false;
}

// ============================================================================
// Parameter Helpers
// ============================================================================

svm_parameter getDefaultParameter(int svm_type, int kernel_type) {
    svm_parameter param;
    
    param.svm_type = svm_type;
    param.kernel_type = kernel_type;
    param.degree = 3;
    param.gamma = 0.5;
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 100;
    param.C = 1;
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = nullptr;
    param.weight = nullptr;
    
    return param;
}

// ============================================================================
// Dataset Generators
// ============================================================================

std::unique_ptr<SvmProblemBuilder> createLinearlySeperableData(int n_samples, unsigned int seed) {
    auto builder = std::make_unique<SvmProblemBuilder>();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dist(-1.0, 1.0);
    std::normal_distribution<> noise(0, 0.1);
    
    // Class 1: points in upper-right region
    for (int i = 0; i < n_samples; ++i) {
        double x1 = std::abs(dist(gen)) + 0.5 + noise(gen);
        double x2 = std::abs(dist(gen)) + 0.5 + noise(gen);
        builder->addDenseSample(1, {x1, x2});
    }
    
    // Class -1: points in lower-left region
    for (int i = 0; i < n_samples; ++i) {
        double x1 = -std::abs(dist(gen)) - 0.5 + noise(gen);
        double x2 = -std::abs(dist(gen)) - 0.5 + noise(gen);
        builder->addDenseSample(-1, {x1, x2});
    }
    
    return builder;
}

std::unique_ptr<SvmProblemBuilder> createXorData(int n_samples, double noise_level, unsigned int seed) {
    auto builder = std::make_unique<SvmProblemBuilder>();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dist(0.2, 1.0);
    std::normal_distribution<> noise(0, noise_level);
    
    // Class 1: upper-left and lower-right quadrants
    for (int i = 0; i < n_samples; ++i) {
        double x1 = -dist(gen) + noise(gen);
        double x2 = dist(gen) + noise(gen);
        builder->addDenseSample(1, {x1, x2});
    }
    for (int i = 0; i < n_samples; ++i) {
        double x1 = dist(gen) + noise(gen);
        double x2 = -dist(gen) + noise(gen);
        builder->addDenseSample(1, {x1, x2});
    }
    
    // Class -1: upper-right and lower-left quadrants
    for (int i = 0; i < n_samples; ++i) {
        double x1 = dist(gen) + noise(gen);
        double x2 = dist(gen) + noise(gen);
        builder->addDenseSample(-1, {x1, x2});
    }
    for (int i = 0; i < n_samples; ++i) {
        double x1 = -dist(gen) + noise(gen);
        double x2 = -dist(gen) + noise(gen);
        builder->addDenseSample(-1, {x1, x2});
    }
    
    return builder;
}

std::unique_ptr<SvmProblemBuilder> createMultiClassData(int n_classes, int n_samples, 
                                                         int n_features, unsigned int seed) {
    auto builder = std::make_unique<SvmProblemBuilder>();
    std::mt19937 gen(seed);
    std::normal_distribution<> noise(0, 0.3);
    
    for (int c = 0; c < n_classes; ++c) {
        // Create cluster centers
        std::vector<double> center(n_features);
        for (int f = 0; f < n_features; ++f) {
            center[f] = static_cast<double>(c * 3);  // Spread centers apart
            if (f == c % n_features) {
                center[f] += 2.0;  // Make clusters distinguishable
            }
        }
        
        // Generate samples around center
        for (int i = 0; i < n_samples; ++i) {
            std::vector<double> features(n_features);
            for (int f = 0; f < n_features; ++f) {
                features[f] = center[f] + noise(gen);
            }
            builder->addDenseSample(static_cast<double>(c + 1), features);
        }
    }
    
    return builder;
}

std::unique_ptr<SvmProblemBuilder> createRegressionData(int n_samples, double noise_level, unsigned int seed) {
    auto builder = std::make_unique<SvmProblemBuilder>();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> x_dist(-3.0, 3.0);
    std::normal_distribution<> noise(0, noise_level);
    
    for (int i = 0; i < n_samples; ++i) {
        double x1 = x_dist(gen);
        double x2 = x_dist(gen);
        // Target: y = 2*x1 + 3*x2 + 1 + noise
        double y = 2 * x1 + 3 * x2 + 1 + noise(gen);
        builder->addDenseSample(y, {x1, x2});
    }
    
    return builder;
}

std::unique_ptr<SvmProblemBuilder> loadHeartScale(const std::string& filepath) {
    auto builder = std::make_unique<SvmProblemBuilder>();
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return builder;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        double label;
        iss >> label;
        
        std::vector<std::pair<int, double>> features;
        std::string token;
        while (iss >> token) {
            auto colon = token.find(':');
            if (colon != std::string::npos) {
                int idx = std::stoi(token.substr(0, colon));
                double val = std::stod(token.substr(colon + 1));
                features.emplace_back(idx, val);
            }
        }
        
        builder->addSample(label, features);
    }
    
    return builder;
}

// ============================================================================
// Metric Functions
// ============================================================================

double calculateAccuracy(const std::vector<double>& predictions, const std::vector<double>& truth) {
    if (predictions.size() != truth.size() || predictions.empty()) {
        return 0.0;
    }
    
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == truth[i]) {
            ++correct;
        }
    }
    
    return static_cast<double>(correct) / static_cast<double>(predictions.size());
}

double calculateMSE(const std::vector<double>& predictions, const std::vector<double>& truth) {
    if (predictions.size() != truth.size() || predictions.empty()) {
        return std::numeric_limits<double>::max();
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        double diff = predictions[i] - truth[i];
        sum += diff * diff;
    }
    
    return sum / static_cast<double>(predictions.size());
}

bool almostEqual(double a, double b, double epsilon) {
    return std::abs(a - b) < epsilon;
}

// ============================================================================
// Output Control
// ============================================================================

static void nullPrintFunc(const char*) {}

void suppressOutput() {
    svm_set_print_string_function(&nullPrintFunc);
}

void restoreOutput() {
    svm_set_print_string_function(nullptr);
}

// ============================================================================
// File Utilities
// ============================================================================

std::string getTempFilePath(const std::string& suffix) {
    static int counter = 0;
    std::string path = std::filesystem::temp_directory_path().string();
    path += "/libsvm_test_" + std::to_string(counter++) + suffix;
    return path;
}

void deleteTempFile(const std::string& filepath) {
    std::filesystem::remove(filepath);
}

} // namespace libsvm_test
