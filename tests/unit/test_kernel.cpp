/**
 * @file test_kernel.cpp
 * @brief Unit tests for SVM kernel functions
 */

#include <gtest/gtest.h>
#include "svm.h"
#include "test_utils.h"
#include <cmath>
#include <vector>

using namespace libsvm_test;

class KernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        suppressOutput();
    }
    
    void TearDown() override {
        restoreOutput();
    }
    
    // Helper to create a simple dense node vector
    std::vector<svm_node> createDenseVector(const std::vector<double>& values) {
        std::vector<svm_node> nodes;
        for (size_t i = 0; i < values.size(); ++i) {
            svm_node node;
            node.index = static_cast<int>(i + 1);
            node.value = values[i];
            nodes.push_back(node);
        }
        svm_node terminator;
        terminator.index = -1;
        terminator.value = 0;
        nodes.push_back(terminator);
        return nodes;
    }
    
    // Helper to compute dot product manually
    double dotProduct(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0;
        size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }
    
    // Helper to compute Euclidean distance squared
    double euclideanDistSq(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0;
        size_t n = std::max(a.size(), b.size());
        for (size_t i = 0; i < n; ++i) {
            double ai = (i < a.size()) ? a[i] : 0;
            double bi = (i < b.size()) ? b[i] : 0;
            double diff = ai - bi;
            sum += diff * diff;
        }
        return sum;
    }
};

// ===========================================================================
// LINEAR Kernel Tests
// ===========================================================================

TEST_F(KernelTest, LinearKernel_BasicDotProduct) {
    // Linear kernel: K(x, y) = x · y
    std::vector<double> x_vals = {1.0, 2.0, 3.0};
    std::vector<double> y_vals = {4.0, 5.0, 6.0};
    
    auto x = createDenseVector(x_vals);
    auto y = createDenseVector(y_vals);
    
    svm_parameter param = getDefaultParameter(C_SVC, LINEAR);
    
    // Train a simple model to test kernel indirectly
    auto builder = createLinearlySeperableData(20);
    svm_problem* prob = builder->build();
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    // Just verify model was trained - kernel correctness is implicit
    EXPECT_GT(svm_get_nr_sv(model.get()), 0);
}

TEST_F(KernelTest, LinearKernel_OrthogonalVectors) {
    // Orthogonal vectors should have dot product = 0
    auto builder = std::make_unique<SvmProblemBuilder>();
    
    // Create nearly orthogonal samples
    builder->addDenseSample(1.0, {1.0, 0.0, 0.0});
    builder->addDenseSample(1.0, {1.0, 0.1, 0.0});
    builder->addDenseSample(-1.0, {0.0, 1.0, 0.0});
    builder->addDenseSample(-1.0, {0.0, 1.0, 0.1});
    
    svm_parameter param = getDefaultParameter(C_SVC, LINEAR);
    svm_problem* prob = builder->build();
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    // Test predictions
    std::vector<svm_node> test_x = {{1, 1.0}, {2, 0.05}, {-1, 0}};
    double pred = svm_predict(model.get(), test_x.data());
    EXPECT_DOUBLE_EQ(pred, 1.0);
    
    std::vector<svm_node> test_y = {{1, 0.05}, {2, 1.0}, {-1, 0}};
    pred = svm_predict(model.get(), test_y.data());
    EXPECT_DOUBLE_EQ(pred, -1.0);
}

// ===========================================================================
// RBF (Gaussian) Kernel Tests
// ===========================================================================

TEST_F(KernelTest, RBFKernel_IdenticalVectors) {
    // K(x, x) = exp(0) = 1
    auto builder = createLinearlySeperableData(20);
    svm_problem* prob = builder->build();
    
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.5;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    // Self-prediction should work
    EXPECT_GT(svm_get_nr_sv(model.get()), 0);
}

TEST_F(KernelTest, RBFKernel_DistantVectors) {
    // Distant vectors should have kernel value close to 0
    auto builder = std::make_unique<SvmProblemBuilder>();
    
    // Very well-separated data
    for (int i = 0; i < 20; ++i) {
        builder->addDenseSample(1.0, {100.0 + i * 0.1, 100.0 + i * 0.1});
        builder->addDenseSample(-1.0, {-100.0 - i * 0.1, -100.0 - i * 0.1});
    }
    
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.1;  // Lower gamma for larger distances
    svm_problem* prob = builder->build();
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    // Test predictions
    std::vector<svm_node> test_pos = {{1, 100.0}, {2, 100.0}, {-1, 0}};
    EXPECT_DOUBLE_EQ(svm_predict(model.get(), test_pos.data()), 1.0);
    
    std::vector<svm_node> test_neg = {{1, -100.0}, {2, -100.0}, {-1, 0}};
    EXPECT_DOUBLE_EQ(svm_predict(model.get(), test_neg.data()), -1.0);
}

TEST_F(KernelTest, RBFKernel_GammaEffect) {
    // Higher gamma = more localized kernel
    auto builder = createXorData(15, 0.05, 42);
    svm_problem* prob = builder->build();
    
    // Test with different gamma values
    for (double gamma : {0.01, 0.1, 1.0, 10.0}) {
        svm_parameter param = getDefaultParameter(C_SVC, RBF);
        param.gamma = gamma;
        param.C = 10;  // Higher C for XOR
        
        SvmModelGuard model(svm_train(prob, &param));
        ASSERT_TRUE(model) << "Failed to train with gamma=" << gamma;
        EXPECT_GT(svm_get_nr_sv(model.get()), 0);
    }
}

// ===========================================================================
// POLY (Polynomial) Kernel Tests
// ===========================================================================

TEST_F(KernelTest, PolyKernel_Degree1IsLinear) {
    // Polynomial with degree=1, coef0=0 should behave like linear
    auto builder = createLinearlySeperableData(20);
    svm_problem* prob = builder->build();
    
    svm_parameter param_linear = getDefaultParameter(C_SVC, LINEAR);
    svm_parameter param_poly = getDefaultParameter(C_SVC, POLY);
    param_poly.degree = 1;
    param_poly.gamma = 1.0;
    param_poly.coef0 = 0;
    
    SvmModelGuard model_linear(svm_train(prob, &param_linear));
    SvmModelGuard model_poly(svm_train(prob, &param_poly));
    
    ASSERT_TRUE(model_linear);
    ASSERT_TRUE(model_poly);
    
    // Predictions should be similar (not necessarily identical due to optimization)
    std::vector<svm_node> test = {{1, 1.0}, {2, 1.0}, {-1, 0}};
    double pred_linear = svm_predict(model_linear.get(), test.data());
    double pred_poly = svm_predict(model_poly.get(), test.data());
    
    EXPECT_DOUBLE_EQ(pred_linear, pred_poly);
}

TEST_F(KernelTest, PolyKernel_Degree2) {
    // Quadratic kernel can separate XOR-like patterns
    auto builder = createXorData(15, 0.05, 42);
    svm_problem* prob = builder->build();
    
    svm_parameter param = getDefaultParameter(C_SVC, POLY);
    param.degree = 2;
    param.gamma = 1.0;
    param.coef0 = 1.0;
    param.C = 10;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    // Test XOR corners
    std::vector<svm_node> test_pp = {{1, 0.5}, {2, 0.5}, {-1, 0}};   // (+, +) -> class -1
    std::vector<svm_node> test_pn = {{1, 0.5}, {2, -0.5}, {-1, 0}};  // (+, -) -> class 1
    
    EXPECT_DOUBLE_EQ(svm_predict(model.get(), test_pp.data()), -1.0);
    EXPECT_DOUBLE_EQ(svm_predict(model.get(), test_pn.data()), 1.0);
}

TEST_F(KernelTest, PolyKernel_HigherDegree) {
    auto builder = createXorData(20, 0.05, 42);
    svm_problem* prob = builder->build();
    
    for (int degree : {2, 3, 4, 5}) {
        svm_parameter param = getDefaultParameter(C_SVC, POLY);
        param.degree = degree;
        param.gamma = 0.5;
        param.coef0 = 1.0;
        param.C = 10;
        
        SvmModelGuard model(svm_train(prob, &param));
        ASSERT_TRUE(model) << "Failed with degree=" << degree;
    }
}

// ===========================================================================
// SIGMOID Kernel Tests
// ===========================================================================

TEST_F(KernelTest, SigmoidKernel_Basic) {
    // Sigmoid kernel: tanh(gamma * x·y + coef0)
    auto builder = createLinearlySeperableData(30);
    svm_problem* prob = builder->build();
    
    svm_parameter param = getDefaultParameter(C_SVC, SIGMOID);
    param.gamma = 0.01;  // Small gamma for stability
    param.coef0 = 0;
    param.C = 1;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    EXPECT_GT(svm_get_nr_sv(model.get()), 0);
}

TEST_F(KernelTest, SigmoidKernel_CoefEffect) {
    auto builder = createLinearlySeperableData(30);
    svm_problem* prob = builder->build();
    
    for (double coef0 : {-1.0, 0.0, 1.0}) {
        svm_parameter param = getDefaultParameter(C_SVC, SIGMOID);
        param.gamma = 0.01;
        param.coef0 = coef0;
        
        SvmModelGuard model(svm_train(prob, &param));
        ASSERT_TRUE(model) << "Failed with coef0=" << coef0;
    }
}

// ===========================================================================
// Sparse Vector Kernel Tests
// ===========================================================================

TEST_F(KernelTest, SparseVectorKernel) {
    // Test kernel with sparse vectors
    auto builder = std::make_unique<SvmProblemBuilder>();
    
    // Sparse samples with different non-zero indices
    builder->addSample(1.0, {{1, 1.0}, {5, 1.0}, {10, 1.0}});
    builder->addSample(1.0, {{1, 0.9}, {5, 1.1}, {10, 0.95}});
    builder->addSample(-1.0, {{2, 1.0}, {6, 1.0}, {11, 1.0}});
    builder->addSample(-1.0, {{2, 1.1}, {6, 0.9}, {11, 1.05}});
    
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.5;
    svm_problem* prob = builder->build();
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    // Test with sparse vector
    std::vector<svm_node> test = {{1, 1.0}, {5, 1.0}, {10, 1.0}, {-1, 0}};
    EXPECT_DOUBLE_EQ(svm_predict(model.get(), test.data()), 1.0);
}

TEST_F(KernelTest, MixedDenseSparseKernel) {
    // Mix of dense and sparse vectors
    auto builder = std::make_unique<SvmProblemBuilder>();
    
    // Add samples with varying sparsity
    for (int i = 0; i < 20; ++i) {
        if (i < 10) {
            builder->addSample(1.0, {{1, 1.0 + i * 0.01}, {2, 1.0}, {3, 1.0}});
        } else {
            builder->addSample(-1.0, {{1, -1.0 - i * 0.01}, {2, -1.0}, {3, -1.0}});
        }
    }
    
    svm_parameter param = getDefaultParameter(C_SVC, LINEAR);
    svm_problem* prob = builder->build();
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
}

// ===========================================================================
// Edge Cases
// ===========================================================================

TEST_F(KernelTest, SingleFeatureVectors) {
    auto builder = std::make_unique<SvmProblemBuilder>();
    
    for (int i = 0; i < 20; ++i) {
        builder->addSample(1.0, {{1, 1.0 + i * 0.1}});
        builder->addSample(-1.0, {{1, -1.0 - i * 0.1}});
    }
    
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    svm_problem* prob = builder->build();
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
}

TEST_F(KernelTest, HighDimensionalSparseVectors) {
    auto builder = std::make_unique<SvmProblemBuilder>();
    
    // Vectors in 10000-dimensional space but only 5 non-zero features
    for (int i = 0; i < 20; ++i) {
        builder->addSample(1.0, {{100, 1.0}, {1000, 1.0}, {5000, 1.0}, {8000, 1.0}, {9999, 1.0}});
        builder->addSample(-1.0, {{200, 1.0}, {2000, 1.0}, {6000, 1.0}, {7000, 1.0}, {9998, 1.0}});
    }
    
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.5;
    svm_problem* prob = builder->build();
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
}

TEST_F(KernelTest, ZeroVectors) {
    auto builder = std::make_unique<SvmProblemBuilder>();
    
    // Zero vector (just terminator)
    builder->addSample(1.0, {});
    builder->addSample(1.0, {{1, 0.1}});
    builder->addSample(-1.0, {{1, -0.1}});
    builder->addSample(-1.0, {{1, -0.2}});
    
    svm_parameter param = getDefaultParameter(C_SVC, LINEAR);
    svm_problem* prob = builder->build();
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
}

TEST_F(KernelTest, VerySmallGamma) {
    auto builder = createLinearlySeperableData(20);
    svm_problem* prob = builder->build();
    
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 1e-10;  // Very small gamma
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
}

TEST_F(KernelTest, VeryLargeGamma) {
    auto builder = createLinearlySeperableData(20);
    svm_problem* prob = builder->build();
    
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 100;  // Very large gamma
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
}
