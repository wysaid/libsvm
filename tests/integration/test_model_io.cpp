/**
 * @file test_model_io.cpp
 * @brief Integration tests for SVM model save/load operations
 */

#include <gtest/gtest.h>
#include "svm.h"
#include "test_utils.h"
#include <vector>
#include <cmath>
#include <fstream>

using namespace libsvm_test;

class ModelIOTest : public ::testing::Test {
protected:
    void SetUp() override {
        suppressOutput();
    }
    
    void TearDown() override {
        restoreOutput();
        // Cleanup temp files
        for (const auto& file : temp_files_) {
            deleteTempFile(file);
        }
    }
    
    std::string createTempFile(const std::string& suffix = ".model") {
        std::string path = getTempFilePath(suffix);
        temp_files_.push_back(path);
        return path;
    }
    
private:
    std::vector<std::string> temp_files_;
};

// ===========================================================================
// Basic Save/Load Tests
// ===========================================================================

TEST_F(ModelIOTest, SaveAndLoadModel) {
    auto builder = createLinearlySeperableData(30, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    std::string model_path = createTempFile();
    
    int save_result = svm_save_model(model_path.c_str(), model.get());
    EXPECT_EQ(save_result, 0) << "Failed to save model";
    
    SvmModelGuard loaded_model(svm_load_model(model_path.c_str()));
    ASSERT_TRUE(loaded_model) << "Failed to load model";
    
    // Verify loaded model properties
    EXPECT_EQ(svm_get_svm_type(loaded_model.get()), svm_get_svm_type(model.get()));
    EXPECT_EQ(svm_get_nr_class(loaded_model.get()), svm_get_nr_class(model.get()));
    EXPECT_EQ(svm_get_nr_sv(loaded_model.get()), svm_get_nr_sv(model.get()));
}

TEST_F(ModelIOTest, LoadedModelPredictions) {
    auto builder = createLinearlySeperableData(30, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    // Get predictions from original model
    std::vector<double> original_preds(prob->l);
    for (int i = 0; i < prob->l; ++i) {
        original_preds[i] = svm_predict(model.get(), prob->x[i]);
    }
    
    // Save and load
    std::string model_path = createTempFile();
    svm_save_model(model_path.c_str(), model.get());
    SvmModelGuard loaded_model(svm_load_model(model_path.c_str()));
    ASSERT_TRUE(loaded_model);
    
    // Predictions should be identical
    for (int i = 0; i < prob->l; ++i) {
        double loaded_pred = svm_predict(loaded_model.get(), prob->x[i]);
        EXPECT_DOUBLE_EQ(loaded_pred, original_preds[i]) << "Mismatch at sample " << i;
    }
}

TEST_F(ModelIOTest, LoadedModelDecisionValues) {
    auto builder = createLinearlySeperableData(30, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    std::string model_path = createTempFile();
    svm_save_model(model_path.c_str(), model.get());
    SvmModelGuard loaded_model(svm_load_model(model_path.c_str()));
    ASSERT_TRUE(loaded_model);
    
    // Compare decision values
    for (int i = 0; i < std::min(5, prob->l); ++i) {
        double dv_orig, dv_loaded;
        svm_predict_values(model.get(), prob->x[i], &dv_orig);
        svm_predict_values(loaded_model.get(), prob->x[i], &dv_loaded);
        EXPECT_NEAR(dv_orig, dv_loaded, 1e-6) << "Mismatch at sample " << i;
    }
}

// ===========================================================================
// Different SVM Type Tests
// ===========================================================================

TEST_F(ModelIOTest, SaveLoadC_SVC) {
    auto builder = createLinearlySeperableData(30, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    SvmModelGuard model(svm_train(prob, &param));
    std::string model_path = createTempFile();
    
    svm_save_model(model_path.c_str(), model.get());
    SvmModelGuard loaded_model(svm_load_model(model_path.c_str()));
    
    EXPECT_EQ(svm_get_svm_type(loaded_model.get()), C_SVC);
}

TEST_F(ModelIOTest, SaveLoadNU_SVC) {
    auto builder = createLinearlySeperableData(30, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(NU_SVC, RBF);
    param.nu = 0.5;
    
    SvmModelGuard model(svm_train(prob, &param));
    std::string model_path = createTempFile();
    
    svm_save_model(model_path.c_str(), model.get());
    SvmModelGuard loaded_model(svm_load_model(model_path.c_str()));
    
    EXPECT_EQ(svm_get_svm_type(loaded_model.get()), NU_SVC);
}

TEST_F(ModelIOTest, SaveLoadONE_CLASS) {
    auto builder = std::make_unique<SvmProblemBuilder>();
    for (int i = 0; i < 50; ++i) {
        builder->addDenseSample(1.0, {0.5 + i * 0.01, 0.5 + i * 0.01});
    }
    
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(ONE_CLASS, RBF);
    param.nu = 0.1;
    
    SvmModelGuard model(svm_train(prob, &param));
    std::string model_path = createTempFile();
    
    svm_save_model(model_path.c_str(), model.get());
    SvmModelGuard loaded_model(svm_load_model(model_path.c_str()));
    
    EXPECT_EQ(svm_get_svm_type(loaded_model.get()), ONE_CLASS);
}

TEST_F(ModelIOTest, SaveLoadEPSILON_SVR) {
    auto builder = createRegressionData(50, 0.1, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(EPSILON_SVR, RBF);
    param.p = 0.1;
    
    SvmModelGuard model(svm_train(prob, &param));
    std::string model_path = createTempFile();
    
    svm_save_model(model_path.c_str(), model.get());
    SvmModelGuard loaded_model(svm_load_model(model_path.c_str()));
    
    EXPECT_EQ(svm_get_svm_type(loaded_model.get()), EPSILON_SVR);
    
    // Verify regression predictions
    for (int i = 0; i < std::min(5, prob->l); ++i) {
        double orig_pred = svm_predict(model.get(), prob->x[i]);
        double loaded_pred = svm_predict(loaded_model.get(), prob->x[i]);
        EXPECT_NEAR(orig_pred, loaded_pred, 1e-6);
    }
}

TEST_F(ModelIOTest, SaveLoadNU_SVR) {
    auto builder = createRegressionData(50, 0.1, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(NU_SVR, RBF);
    param.nu = 0.5;
    
    SvmModelGuard model(svm_train(prob, &param));
    std::string model_path = createTempFile();
    
    svm_save_model(model_path.c_str(), model.get());
    SvmModelGuard loaded_model(svm_load_model(model_path.c_str()));
    
    EXPECT_EQ(svm_get_svm_type(loaded_model.get()), NU_SVR);
}

// ===========================================================================
// Different Kernel Type Tests
// ===========================================================================

TEST_F(ModelIOTest, SaveLoadLinearKernel) {
    auto builder = createLinearlySeperableData(30, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, LINEAR);
    
    SvmModelGuard model(svm_train(prob, &param));
    std::string model_path = createTempFile();
    
    svm_save_model(model_path.c_str(), model.get());
    SvmModelGuard loaded_model(svm_load_model(model_path.c_str()));
    
    EXPECT_EQ(loaded_model->param.kernel_type, LINEAR);
}

TEST_F(ModelIOTest, SaveLoadPolynomialKernel) {
    auto builder = createXorData(20, 0.05, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, POLY);
    param.degree = 3;
    param.gamma = 0.5;
    param.coef0 = 1.0;
    
    SvmModelGuard model(svm_train(prob, &param));
    std::string model_path = createTempFile();
    
    svm_save_model(model_path.c_str(), model.get());
    SvmModelGuard loaded_model(svm_load_model(model_path.c_str()));
    
    EXPECT_EQ(loaded_model->param.kernel_type, POLY);
    EXPECT_EQ(loaded_model->param.degree, 3);
    EXPECT_DOUBLE_EQ(loaded_model->param.gamma, 0.5);
    EXPECT_DOUBLE_EQ(loaded_model->param.coef0, 1.0);
}

TEST_F(ModelIOTest, SaveLoadRBFKernel) {
    auto builder = createLinearlySeperableData(30, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.123;
    
    SvmModelGuard model(svm_train(prob, &param));
    std::string model_path = createTempFile();
    
    svm_save_model(model_path.c_str(), model.get());
    SvmModelGuard loaded_model(svm_load_model(model_path.c_str()));
    
    EXPECT_EQ(loaded_model->param.kernel_type, RBF);
    EXPECT_DOUBLE_EQ(loaded_model->param.gamma, 0.123);
}

TEST_F(ModelIOTest, SaveLoadSigmoidKernel) {
    auto builder = createLinearlySeperableData(30, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, SIGMOID);
    param.gamma = 0.01;
    param.coef0 = 0;
    
    SvmModelGuard model(svm_train(prob, &param));
    std::string model_path = createTempFile();
    
    svm_save_model(model_path.c_str(), model.get());
    SvmModelGuard loaded_model(svm_load_model(model_path.c_str()));
    
    EXPECT_EQ(loaded_model->param.kernel_type, SIGMOID);
}

// ===========================================================================
// Multi-class Model Tests
// ===========================================================================

TEST_F(ModelIOTest, SaveLoadMultiClassModel) {
    auto builder = createMultiClassData(4, 30, 4, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    SvmModelGuard model(svm_train(prob, &param));
    std::string model_path = createTempFile();
    
    svm_save_model(model_path.c_str(), model.get());
    SvmModelGuard loaded_model(svm_load_model(model_path.c_str()));
    
    EXPECT_EQ(svm_get_nr_class(loaded_model.get()), 4);
    
    // Verify labels are preserved
    std::vector<int> orig_labels(4), loaded_labels(4);
    svm_get_labels(model.get(), orig_labels.data());
    svm_get_labels(loaded_model.get(), loaded_labels.data());
    
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(orig_labels[i], loaded_labels[i]);
    }
}

// ===========================================================================
// Probability Model Tests
// ===========================================================================

TEST_F(ModelIOTest, SaveLoadProbabilityModel) {
    auto builder = createLinearlySeperableData(50, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.probability = 1;
    
    SvmModelGuard model(svm_train(prob, &param));
    std::string model_path = createTempFile();
    
    svm_save_model(model_path.c_str(), model.get());
    SvmModelGuard loaded_model(svm_load_model(model_path.c_str()));
    
    EXPECT_EQ(svm_check_probability_model(loaded_model.get()), 1);
    
    // Test probability predictions
    std::vector<double> orig_probs(2), loaded_probs(2);
    double orig_pred = svm_predict_probability(model.get(), prob->x[0], orig_probs.data());
    double loaded_pred = svm_predict_probability(loaded_model.get(), prob->x[0], loaded_probs.data());
    
    EXPECT_DOUBLE_EQ(orig_pred, loaded_pred);
    for (int i = 0; i < 2; ++i) {
        EXPECT_NEAR(orig_probs[i], loaded_probs[i], 1e-6);
    }
}

TEST_F(ModelIOTest, SaveLoadSVRProbabilityModel) {
    auto builder = createRegressionData(80, 0.1, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(EPSILON_SVR, RBF);
    param.probability = 1;
    
    SvmModelGuard model(svm_train(prob, &param));
    std::string model_path = createTempFile();
    
    svm_save_model(model_path.c_str(), model.get());
    SvmModelGuard loaded_model(svm_load_model(model_path.c_str()));
    
    EXPECT_EQ(svm_check_probability_model(loaded_model.get()), 1);
    
    double orig_prob = svm_get_svr_probability(model.get());
    double loaded_prob = svm_get_svr_probability(loaded_model.get());
    EXPECT_NEAR(orig_prob, loaded_prob, 1e-6);
}

// ===========================================================================
// Error Handling Tests
// ===========================================================================

TEST_F(ModelIOTest, LoadNonexistentFile) {
    SvmModelGuard loaded_model(svm_load_model("/nonexistent/path/model.txt"));
    EXPECT_FALSE(loaded_model);
}

TEST_F(ModelIOTest, SaveToInvalidPath) {
    auto builder = createLinearlySeperableData(10, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, LINEAR);
    
    SvmModelGuard model(svm_train(prob, &param));
    
    int result = svm_save_model("/nonexistent/directory/model.txt", model.get());
    EXPECT_NE(result, 0);
}

TEST_F(ModelIOTest, LoadCorruptedFile) {
    std::string path = createTempFile();
    
    // Write corrupted content
    std::ofstream file(path);
    file << "invalid model content\n";
    file << "garbage data\n";
    file.close();
    
    SvmModelGuard loaded_model(svm_load_model(path.c_str()));
    // Should either return nullptr or a partially valid model
    // The exact behavior depends on implementation
}

TEST_F(ModelIOTest, LoadEmptyFile) {
    std::string path = createTempFile();
    
    // Create empty file
    std::ofstream file(path);
    file.close();
    
    SvmModelGuard loaded_model(svm_load_model(path.c_str()));
    EXPECT_FALSE(loaded_model);
}

// ===========================================================================
// Edge Cases
// ===========================================================================

TEST_F(ModelIOTest, SaveLoadSparseModel) {
    auto builder = std::make_unique<SvmProblemBuilder>();
    
    // Sparse features
    builder->addSample(1.0, {{1, 1.0}, {100, 0.5}, {1000, 0.3}});
    builder->addSample(1.0, {{1, 0.9}, {100, 0.6}, {1000, 0.2}});
    builder->addSample(-1.0, {{2, 1.0}, {200, 0.5}, {2000, 0.3}});
    builder->addSample(-1.0, {{2, 1.1}, {200, 0.4}, {2000, 0.4}});
    
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    SvmModelGuard model(svm_train(prob, &param));
    std::string model_path = createTempFile();
    
    svm_save_model(model_path.c_str(), model.get());
    SvmModelGuard loaded_model(svm_load_model(model_path.c_str()));
    ASSERT_TRUE(loaded_model);
    
    // Verify predictions match
    for (int i = 0; i < prob->l; ++i) {
        double orig = svm_predict(model.get(), prob->x[i]);
        double loaded = svm_predict(loaded_model.get(), prob->x[i]);
        EXPECT_DOUBLE_EQ(orig, loaded);
    }
}

TEST_F(ModelIOTest, MultipleLoadsSameFile) {
    auto builder = createLinearlySeperableData(20, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    SvmModelGuard model(svm_train(prob, &param));
    std::string model_path = createTempFile();
    svm_save_model(model_path.c_str(), model.get());
    
    // Load multiple times
    for (int i = 0; i < 5; ++i) {
        SvmModelGuard loaded(svm_load_model(model_path.c_str()));
        ASSERT_TRUE(loaded) << "Load " << i << " failed";
        
        double pred = svm_predict(loaded.get(), prob->x[0]);
        double expected = svm_predict(model.get(), prob->x[0]);
        EXPECT_DOUBLE_EQ(pred, expected);
    }
}

TEST_F(ModelIOTest, OverwriteExistingModel) {
    auto builder1 = createLinearlySeperableData(20, 42);
    svm_problem* prob1 = builder1->build();
    svm_parameter param1 = getDefaultParameter(C_SVC, LINEAR);
    
    auto builder2 = createXorData(20, 0.05, 123);
    svm_problem* prob2 = builder2->build();
    svm_parameter param2 = getDefaultParameter(C_SVC, RBF);
    
    SvmModelGuard model1(svm_train(prob1, &param1));
    SvmModelGuard model2(svm_train(prob2, &param2));
    
    std::string model_path = createTempFile();
    
    // Save first model
    svm_save_model(model_path.c_str(), model1.get());
    
    // Overwrite with second model
    svm_save_model(model_path.c_str(), model2.get());
    
    // Load should get second model
    SvmModelGuard loaded(svm_load_model(model_path.c_str()));
    ASSERT_TRUE(loaded);
    
    EXPECT_EQ(loaded->param.kernel_type, RBF);
}
