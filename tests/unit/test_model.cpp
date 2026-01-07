/**
 * @file test_model.cpp
 * @brief Unit tests for svm_model structure and related functions
 */

#include <gtest/gtest.h>
#include "svm.h"
#include "test_utils.h"
#include <vector>
#include <cstring>

using namespace libsvm_test;

class SvmModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        suppressOutput();
    }
    
    void TearDown() override {
        restoreOutput();
    }
};

// ===========================================================================
// Model Creation Tests
// ===========================================================================

TEST_F(SvmModelTest, TrainCreatesModel) {
    auto builder = createLinearlySeperableData(20);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter();
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
}

TEST_F(SvmModelTest, ModelHasSupportVectors) {
    auto builder = createLinearlySeperableData(20);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter();
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    int nr_sv = svm_get_nr_sv(model.get());
    EXPECT_GT(nr_sv, 0);
    EXPECT_LE(nr_sv, prob->l);
}

TEST_F(SvmModelTest, ModelHasValidSvmType) {
    auto builder = createLinearlySeperableData(20);
    svm_problem* prob = builder->build();
    
    int svm_types[] = {C_SVC, NU_SVC};
    for (int st : svm_types) {
        svm_parameter param = getDefaultParameter(st);
        SvmModelGuard model(svm_train(prob, &param));
        ASSERT_TRUE(model);
        EXPECT_EQ(svm_get_svm_type(model.get()), st);
    }
}

// ===========================================================================
// Model Query Tests
// ===========================================================================

TEST_F(SvmModelTest, GetNrClass) {
    auto builder = createLinearlySeperableData(20);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC);
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    int nr_class = svm_get_nr_class(model.get());
    EXPECT_EQ(nr_class, 2);
}

TEST_F(SvmModelTest, GetNrClassMultiClass) {
    auto builder = createMultiClassData(4, 20, 3);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC);
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    int nr_class = svm_get_nr_class(model.get());
    EXPECT_EQ(nr_class, 4);
}

TEST_F(SvmModelTest, GetLabels) {
    auto builder = createLinearlySeperableData(20);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC);
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    int nr_class = svm_get_nr_class(model.get());
    std::vector<int> labels(nr_class);
    svm_get_labels(model.get(), labels.data());
    
    // Labels should be 1 and -1
    bool has_pos = false, has_neg = false;
    for (int label : labels) {
        if (label == 1) has_pos = true;
        if (label == -1) has_neg = true;
    }
    EXPECT_TRUE(has_pos);
    EXPECT_TRUE(has_neg);
}

TEST_F(SvmModelTest, GetSvIndices) {
    auto builder = createLinearlySeperableData(20);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC);
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    int nr_sv = svm_get_nr_sv(model.get());
    std::vector<int> sv_indices(nr_sv);
    svm_get_sv_indices(model.get(), sv_indices.data());
    
    // SV indices should be 1-based and within training data range
    for (int idx : sv_indices) {
        EXPECT_GE(idx, 1);
        EXPECT_LE(idx, prob->l);
    }
}

// ===========================================================================
// Model for Regression Tests
// ===========================================================================

TEST_F(SvmModelTest, RegressionModel_EpsilonSVR) {
    auto builder = createRegressionData(50, 0.1);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(EPSILON_SVR);
    param.p = 0.1;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    EXPECT_EQ(svm_get_svm_type(model.get()), EPSILON_SVR);
    EXPECT_EQ(svm_get_nr_class(model.get()), 2);  // Always 2 for regression
}

TEST_F(SvmModelTest, RegressionModel_NuSVR) {
    auto builder = createRegressionData(50, 0.1);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(NU_SVR);
    param.nu = 0.5;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    EXPECT_EQ(svm_get_svm_type(model.get()), NU_SVR);
}

// ===========================================================================
// One-Class SVM Tests
// ===========================================================================

TEST_F(SvmModelTest, OneClassModel) {
    auto builder = std::make_unique<SvmProblemBuilder>();
    
    // Generate normal data (no labels needed for one-class)
    for (int i = 0; i < 50; ++i) {
        builder->addDenseSample(1.0, {0.5 + i * 0.01, 0.5 + i * 0.005});
    }
    
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(ONE_CLASS);
    param.nu = 0.1;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    EXPECT_EQ(svm_get_svm_type(model.get()), ONE_CLASS);
}

// ===========================================================================
// Model with Probability Estimates Tests
// ===========================================================================

TEST_F(SvmModelTest, ProbabilityModel) {
    auto builder = createLinearlySeperableData(30);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC);
    param.probability = 1;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    EXPECT_EQ(svm_check_probability_model(model.get()), 1);
}

TEST_F(SvmModelTest, NonProbabilityModel) {
    auto builder = createLinearlySeperableData(20);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC);
    param.probability = 0;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    EXPECT_EQ(svm_check_probability_model(model.get()), 0);
}

TEST_F(SvmModelTest, SVRProbabilityModel) {
    auto builder = createRegressionData(60, 0.1);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(EPSILON_SVR);
    param.probability = 1;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    double svr_prob = svm_get_svr_probability(model.get());
    EXPECT_GT(svr_prob, 0);
}

// ===========================================================================
// Model Memory Management Tests
// ===========================================================================

TEST_F(SvmModelTest, FreeModelContent) {
    auto builder = createLinearlySeperableData(20);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter();
    
    svm_model* model = svm_train(prob, &param);
    ASSERT_NE(model, nullptr);
    
    // Free content but not the model struct itself
    svm_free_model_content(model);
    
    // Model struct still exists but content is freed
    free(model);
    
    // No assertion - just verify no crash
    SUCCEED();
}

TEST_F(SvmModelTest, FreeAndDestroyModel) {
    auto builder = createLinearlySeperableData(20);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter();
    
    svm_model* model = svm_train(prob, &param);
    ASSERT_NE(model, nullptr);
    
    svm_free_and_destroy_model(&model);
    EXPECT_EQ(model, nullptr);
}

TEST_F(SvmModelTest, FreeAndDestroyNullModel) {
    svm_model* model = nullptr;
    
    // Should handle null gracefully
    svm_free_and_destroy_model(&model);
    EXPECT_EQ(model, nullptr);
}

// ===========================================================================
// Model Parameter Preservation Tests
// ===========================================================================

TEST_F(SvmModelTest, ModelPreservesKernelType) {
    auto builder = createLinearlySeperableData(20);
    svm_problem* prob = builder->build();
    
    int kernel_types[] = {LINEAR, POLY, RBF, SIGMOID};
    for (int kt : kernel_types) {
        svm_parameter param = getDefaultParameter(C_SVC, kt);
        SvmModelGuard model(svm_train(prob, &param));
        ASSERT_TRUE(model);
        
        // Verify kernel type is preserved in model
        EXPECT_EQ(model->param.kernel_type, kt);
    }
}

TEST_F(SvmModelTest, ModelPreservesGamma) {
    auto builder = createLinearlySeperableData(20);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.123;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    EXPECT_DOUBLE_EQ(model->param.gamma, 0.123);
}

// ===========================================================================
// Model Structure Validation Tests
// ===========================================================================

TEST_F(SvmModelTest, ModelSVCoefValid) {
    auto builder = createLinearlySeperableData(20);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC);
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    // sv_coef should be allocated
    EXPECT_NE(model->sv_coef, nullptr);
    
    // For 2-class, sv_coef[0] should exist
    int nr_class = svm_get_nr_class(model.get());
    EXPECT_NE(model->sv_coef[0], nullptr);
}

TEST_F(SvmModelTest, ModelRhoValid) {
    auto builder = createLinearlySeperableData(20);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC);
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    // rho should be allocated and valid
    EXPECT_NE(model->rho, nullptr);
}

TEST_F(SvmModelTest, ModelNSVPerClass) {
    auto builder = createLinearlySeperableData(20);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC);
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    int nr_class = svm_get_nr_class(model.get());
    int total_sv = 0;
    for (int i = 0; i < nr_class; ++i) {
        EXPECT_GE(model->nSV[i], 0);
        total_sv += model->nSV[i];
    }
    
    EXPECT_EQ(total_sv, svm_get_nr_sv(model.get()));
}

// ===========================================================================
// Edge Cases
// ===========================================================================

TEST_F(SvmModelTest, SmallDataset) {
    auto builder = std::make_unique<SvmProblemBuilder>();
    
    builder->addDenseSample(1.0, {1.0, 1.0});
    builder->addDenseSample(-1.0, {-1.0, -1.0});
    
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, LINEAR);
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    EXPECT_LE(svm_get_nr_sv(model.get()), 2);
}

TEST_F(SvmModelTest, LargeC) {
    auto builder = createLinearlySeperableData(20);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC);
    param.C = 1e10;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
}

TEST_F(SvmModelTest, SmallC) {
    auto builder = createLinearlySeperableData(20);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC);
    param.C = 1e-10;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
}
