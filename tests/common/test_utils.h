/**
 * @file test_utils.h
 * @brief Common utilities for LibSVM tests
 */

#ifndef LIBSVM_TEST_UTILS_H
#define LIBSVM_TEST_UTILS_H

#include "svm.h"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <memory>
#include <random>

namespace libsvm_test {

/**
 * @brief RAII wrapper for svm_model
 */
class SvmModelGuard {
public:
    explicit SvmModelGuard(svm_model* model = nullptr) : model_(model) {}
    ~SvmModelGuard() {
        if (model_) {
            svm_free_and_destroy_model(&model_);
        }
    }
    
    // Non-copyable
    SvmModelGuard(const SvmModelGuard&) = delete;
    SvmModelGuard& operator=(const SvmModelGuard&) = delete;
    
    // Movable
    SvmModelGuard(SvmModelGuard&& other) noexcept : model_(other.model_) {
        other.model_ = nullptr;
    }
    SvmModelGuard& operator=(SvmModelGuard&& other) noexcept {
        if (this != &other) {
            if (model_) svm_free_and_destroy_model(&model_);
            model_ = other.model_;
            other.model_ = nullptr;
        }
        return *this;
    }
    
    svm_model* get() const { return model_; }
    svm_model* release() {
        svm_model* tmp = model_;
        model_ = nullptr;
        return tmp;
    }
    void reset(svm_model* model = nullptr) {
        if (model_) svm_free_and_destroy_model(&model_);
        model_ = model;
    }
    
    explicit operator bool() const { return model_ != nullptr; }
    svm_model* operator->() const { return model_; }
    
private:
    svm_model* model_;
};

/**
 * @brief Helper class to manage svm_problem and its resources
 */
class SvmProblemBuilder {
public:
    SvmProblemBuilder() = default;
    ~SvmProblemBuilder();
    
    // Non-copyable
    SvmProblemBuilder(const SvmProblemBuilder&) = delete;
    SvmProblemBuilder& operator=(const SvmProblemBuilder&) = delete;
    
    /**
     * @brief Add a sample to the problem
     * @param label Class label
     * @param features Sparse feature vector (index, value pairs)
     */
    void addSample(double label, const std::vector<std::pair<int, double>>& features);
    
    /**
     * @brief Add a sample with dense features
     * @param label Class label  
     * @param features Dense feature vector (values for features 1..n)
     */
    void addDenseSample(double label, const std::vector<double>& features);
    
    /**
     * @brief Build the svm_problem structure
     * @return Pointer to svm_problem (owned by this builder)
     */
    svm_problem* build();
    
    /**
     * @brief Get number of samples added
     */
    int size() const { return static_cast<int>(labels_.size()); }
    
    /**
     * @brief Clear all samples
     */
    void clear();
    
private:
    std::vector<double> labels_;
    std::vector<std::vector<svm_node>> node_vectors_;
    std::vector<svm_node*> x_ptrs_;
    
    svm_problem problem_{};
    bool built_ = false;
};

/**
 * @brief Get default SVM parameters for testing
 */
svm_parameter getDefaultParameter(int svm_type = C_SVC, int kernel_type = RBF);

/**
 * @brief Create a simple linearly separable dataset
 * @param n_samples Number of samples per class
 * @param seed Random seed
 * @return SvmProblemBuilder with generated data
 */
std::unique_ptr<SvmProblemBuilder> createLinearlySeperableData(int n_samples = 50, unsigned int seed = 42);

/**
 * @brief Create XOR dataset (non-linearly separable)
 * @param n_samples Number of samples per quadrant
 * @param noise Noise level
 * @param seed Random seed
 * @return SvmProblemBuilder with generated data
 */
std::unique_ptr<SvmProblemBuilder> createXorData(int n_samples = 25, double noise = 0.1, unsigned int seed = 42);

/**
 * @brief Create multi-class dataset
 * @param n_classes Number of classes
 * @param n_samples Number of samples per class
 * @param n_features Number of features
 * @param seed Random seed
 * @return SvmProblemBuilder with generated data
 */
std::unique_ptr<SvmProblemBuilder> createMultiClassData(int n_classes = 3, int n_samples = 30, 
                                                         int n_features = 4, unsigned int seed = 42);

/**
 * @brief Create regression dataset
 * @param n_samples Number of samples
 * @param noise Noise level
 * @param seed Random seed
 * @return SvmProblemBuilder with generated data
 */
std::unique_ptr<SvmProblemBuilder> createRegressionData(int n_samples = 100, double noise = 0.1, unsigned int seed = 42);

/**
 * @brief Load heart_scale dataset
 * @param filepath Path to heart_scale file
 * @return SvmProblemBuilder with loaded data
 */
std::unique_ptr<SvmProblemBuilder> loadHeartScale(const std::string& filepath);

/**
 * @brief Calculate accuracy between predictions and ground truth
 */
double calculateAccuracy(const std::vector<double>& predictions, const std::vector<double>& truth);

/**
 * @brief Calculate Mean Squared Error for regression
 */
double calculateMSE(const std::vector<double>& predictions, const std::vector<double>& truth);

/**
 * @brief Compare two doubles with tolerance
 */
bool almostEqual(double a, double b, double epsilon = 1e-6);

/**
 * @brief Suppress SVM library output
 */
void suppressOutput();

/**
 * @brief Restore SVM library output
 */
void restoreOutput();

/**
 * @brief Get temporary file path for testing
 */
std::string getTempFilePath(const std::string& suffix = ".model");

/**
 * @brief Delete temporary file
 */
void deleteTempFile(const std::string& filepath);

} // namespace libsvm_test

#endif // LIBSVM_TEST_UTILS_H
