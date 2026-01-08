/**
 * @file test_svm_parameter.cpp
 * @brief Unit tests for svm_parameter structure and validation
 */

#include <gtest/gtest.h>
#include "svm.h"
#include "test_utils.h"

using namespace libsvm_test;

class SvmParameterTest : public ::testing::Test {
protected:
    void SetUp() override {
        suppressOutput();
    }
    
    void TearDown() override {
        restoreOutput();
    }
};

// Test default parameter values
TEST_F(SvmParameterTest, DefaultParameters) {
    svm_parameter param = getDefaultParameter();
    
    EXPECT_EQ(param.svm_type, C_SVC);
    EXPECT_EQ(param.kernel_type, RBF);
    EXPECT_EQ(param.degree, 3);
    EXPECT_DOUBLE_EQ(param.gamma, 0.5);
    EXPECT_DOUBLE_EQ(param.coef0, 0.0);
    EXPECT_DOUBLE_EQ(param.cache_size, 100);
    EXPECT_DOUBLE_EQ(param.eps, 1e-3);
    EXPECT_DOUBLE_EQ(param.C, 1.0);
    EXPECT_DOUBLE_EQ(param.nu, 0.5);
    EXPECT_DOUBLE_EQ(param.p, 0.1);
    EXPECT_EQ(param.shrinking, 1);
    EXPECT_EQ(param.probability, 0);
    EXPECT_EQ(param.nr_weight, 0);
    EXPECT_EQ(param.weight_label, nullptr);
    EXPECT_EQ(param.weight, nullptr);
}

// Test svm_type enumeration values
TEST_F(SvmParameterTest, SvmTypeEnum) {
    EXPECT_EQ(C_SVC, 0);
    EXPECT_EQ(NU_SVC, 1);
    EXPECT_EQ(ONE_CLASS, 2);
    EXPECT_EQ(EPSILON_SVR, 3);
    EXPECT_EQ(NU_SVR, 4);
}

// Test kernel_type enumeration values
TEST_F(SvmParameterTest, KernelTypeEnum) {
    EXPECT_EQ(LINEAR, 0);
    EXPECT_EQ(POLY, 1);
    EXPECT_EQ(RBF, 2);
    EXPECT_EQ(SIGMOID, 3);
    EXPECT_EQ(PRECOMPUTED, 4);
}

// Test parameter validation with valid C_SVC parameters
TEST_F(SvmParameterTest, ValidC_SVCParameters) {
    auto builder = createLinearlySeperableData(10);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    const char* error = svm_check_parameter(prob, &param);
    EXPECT_EQ(error, nullptr) << "Error: " << (error ? error : "");
}

// Test parameter validation with valid NU_SVC parameters
TEST_F(SvmParameterTest, ValidNU_SVCParameters) {
    auto builder = createLinearlySeperableData(10);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(NU_SVC, RBF);
    param.nu = 0.5;
    
    const char* error = svm_check_parameter(prob, &param);
    EXPECT_EQ(error, nullptr) << "Error: " << (error ? error : "");
}

// Test parameter validation with valid ONE_CLASS parameters
TEST_F(SvmParameterTest, ValidOneClassParameters) {
    auto builder = createLinearlySeperableData(10);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(ONE_CLASS, RBF);
    param.nu = 0.1;
    
    const char* error = svm_check_parameter(prob, &param);
    EXPECT_EQ(error, nullptr) << "Error: " << (error ? error : "");
}

// Test parameter validation with valid EPSILON_SVR parameters
TEST_F(SvmParameterTest, ValidEpsilonSVRParameters) {
    auto builder = createRegressionData(20);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(EPSILON_SVR, RBF);
    param.p = 0.1;
    
    const char* error = svm_check_parameter(prob, &param);
    EXPECT_EQ(error, nullptr) << "Error: " << (error ? error : "");
}

// Test parameter validation with valid NU_SVR parameters
TEST_F(SvmParameterTest, ValidNuSVRParameters) {
    auto builder = createRegressionData(20);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(NU_SVR, RBF);
    param.nu = 0.5;
    
    const char* error = svm_check_parameter(prob, &param);
    EXPECT_EQ(error, nullptr) << "Error: " << (error ? error : "");
}

// Test parameter validation with invalid svm_type
TEST_F(SvmParameterTest, InvalidSvmType) {
    auto builder = createLinearlySeperableData(10);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter();
    param.svm_type = 999;  // Invalid
    
    const char* error = svm_check_parameter(prob, &param);
    EXPECT_NE(error, nullptr);
}

// Test parameter validation with invalid kernel_type
TEST_F(SvmParameterTest, InvalidKernelType) {
    auto builder = createLinearlySeperableData(10);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter();
    param.kernel_type = 999;  // Invalid
    
    const char* error = svm_check_parameter(prob, &param);
    EXPECT_NE(error, nullptr);
}

// Test parameter validation with invalid gamma
TEST_F(SvmParameterTest, InvalidGamma) {
    auto builder = createLinearlySeperableData(10);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter();
    param.gamma = -1.0;  // Invalid: must be >= 0
    
    const char* error = svm_check_parameter(prob, &param);
    EXPECT_NE(error, nullptr);
}

// Test parameter validation with invalid cache_size
TEST_F(SvmParameterTest, InvalidCacheSize) {
    auto builder = createLinearlySeperableData(10);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter();
    param.cache_size = 0;  // Invalid: must be > 0
    
    const char* error = svm_check_parameter(prob, &param);
    EXPECT_NE(error, nullptr);
}

// Test parameter validation with invalid eps
TEST_F(SvmParameterTest, InvalidEps) {
    auto builder = createLinearlySeperableData(10);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter();
    param.eps = 0;  // Invalid: must be > 0
    
    const char* error = svm_check_parameter(prob, &param);
    EXPECT_NE(error, nullptr);
}

// Test parameter validation with invalid C for C_SVC
TEST_F(SvmParameterTest, InvalidC) {
    auto builder = createLinearlySeperableData(10);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC);
    param.C = 0;  // Invalid: must be > 0
    
    const char* error = svm_check_parameter(prob, &param);
    EXPECT_NE(error, nullptr);
}

// Test parameter validation with invalid nu for NU_SVC
TEST_F(SvmParameterTest, InvalidNuTooHigh) {
    auto builder = createLinearlySeperableData(10);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(NU_SVC);
    param.nu = 1.5;  // Invalid: must be in (0, 1]
    
    const char* error = svm_check_parameter(prob, &param);
    EXPECT_NE(error, nullptr);
}

// Test parameter validation with invalid nu (too low)
TEST_F(SvmParameterTest, InvalidNuTooLow) {
    auto builder = createLinearlySeperableData(10);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(NU_SVC);
    param.nu = 0;  // Invalid: must be > 0
    
    const char* error = svm_check_parameter(prob, &param);
    EXPECT_NE(error, nullptr);
}

// Test parameter validation with invalid p for EPSILON_SVR
TEST_F(SvmParameterTest, InvalidP) {
    auto builder = createRegressionData(20);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(EPSILON_SVR);
    param.p = -0.1;  // Invalid: must be >= 0
    
    const char* error = svm_check_parameter(prob, &param);
    EXPECT_NE(error, nullptr);
}

// Test different kernel types
TEST_F(SvmParameterTest, AllKernelTypes) {
    auto builder = createLinearlySeperableData(10);
    svm_problem* prob = builder->build();
    
    int kernel_types[] = {LINEAR, POLY, RBF, SIGMOID};
    for (int kt : kernel_types) {
        svm_parameter param = getDefaultParameter(C_SVC, kt);
        const char* error = svm_check_parameter(prob, &param);
        EXPECT_EQ(error, nullptr) << "Failed for kernel_type: " << kt;
    }
}

// Test polynomial kernel specific parameters
TEST_F(SvmParameterTest, PolynomialKernelParams) {
    svm_parameter param = getDefaultParameter(C_SVC, POLY);
    param.degree = 2;
    param.gamma = 0.5;
    param.coef0 = 1.0;
    
    auto builder = createLinearlySeperableData(10);
    svm_problem* prob = builder->build();
    
    const char* error = svm_check_parameter(prob, &param);
    EXPECT_EQ(error, nullptr);
}

// Test sigmoid kernel specific parameters
TEST_F(SvmParameterTest, SigmoidKernelParams) {
    svm_parameter param = getDefaultParameter(C_SVC, SIGMOID);
    param.gamma = 0.01;
    param.coef0 = 0;
    
    auto builder = createLinearlySeperableData(10);
    svm_problem* prob = builder->build();
    
    const char* error = svm_check_parameter(prob, &param);
    EXPECT_EQ(error, nullptr);
}

// Test class weights for C_SVC
TEST_F(SvmParameterTest, ClassWeights) {
    svm_parameter param = getDefaultParameter(C_SVC);
    
    int weight_labels[] = {1, -1};
    double weights[] = {2.0, 1.0};
    
    param.nr_weight = 2;
    param.weight_label = weight_labels;
    param.weight = weights;
    
    auto builder = createLinearlySeperableData(10);
    svm_problem* prob = builder->build();
    
    const char* error = svm_check_parameter(prob, &param);
    EXPECT_EQ(error, nullptr);
    
    // Cleanup
    param.weight_label = nullptr;
    param.weight = nullptr;
}

// Test probability estimation flag
TEST_F(SvmParameterTest, ProbabilityEstimation) {
    svm_parameter param = getDefaultParameter(C_SVC);
    param.probability = 1;
    
    auto builder = createLinearlySeperableData(20);
    svm_problem* prob = builder->build();
    
    const char* error = svm_check_parameter(prob, &param);
    EXPECT_EQ(error, nullptr);
}

// Test shrinking heuristics flag
TEST_F(SvmParameterTest, ShrinkingHeuristics) {
    svm_parameter param = getDefaultParameter();
    
    // Test with shrinking enabled
    param.shrinking = 1;
    auto builder = createLinearlySeperableData(10);
    svm_problem* prob = builder->build();
    
    const char* error = svm_check_parameter(prob, &param);
    EXPECT_EQ(error, nullptr);
    
    // Test with shrinking disabled
    param.shrinking = 0;
    error = svm_check_parameter(prob, &param);
    EXPECT_EQ(error, nullptr);
}
