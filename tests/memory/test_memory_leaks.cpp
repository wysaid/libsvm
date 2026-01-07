/**
 * @file test_memory_leaks.cpp
 * @brief Memory leak detection tests for LibSVM
 * 
 * These tests are designed to be run with AddressSanitizer (ASan) or Valgrind
 * to detect memory leaks. The tests themselves don't assert on memory usage
 * but exercise code paths that could leak memory.
 */

#include <gtest/gtest.h>
#include "svm.h"
#include "test_utils.h"
#include <vector>
#include <string>

using namespace libsvm_test;

class MemoryLeakTest : public ::testing::Test {
protected:
    void SetUp() override {
        suppressOutput();
    }
    
    void TearDown() override {
        restoreOutput();
    }
};

// ===========================================================================
// Model Training/Destruction Memory Tests
// ===========================================================================

TEST_F(MemoryLeakTest, TrainAndFreeModel) {
    auto builder = createLinearlySeperableData(50, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    svm_model* model = svm_train(prob, &param);
    ASSERT_NE(model, nullptr);
    
    svm_free_and_destroy_model(&model);
    EXPECT_EQ(model, nullptr);
}

TEST_F(MemoryLeakTest, RepeatTrainAndFree) {
    // Repeated training and freeing should not leak memory
    for (int iter = 0; iter < 10; ++iter) {
        auto builder = createLinearlySeperableData(30, 42 + iter);
        svm_problem* prob = builder->build();
        svm_parameter param = getDefaultParameter(C_SVC, RBF);
        
        svm_model* model = svm_train(prob, &param);
        ASSERT_NE(model, nullptr);
        svm_free_and_destroy_model(&model);
    }
}

TEST_F(MemoryLeakTest, TrainWithDifferentKernels) {
    auto builder = createLinearlySeperableData(30, 42);
    svm_problem* prob = builder->build();
    
    int kernels[] = {LINEAR, POLY, RBF, SIGMOID};
    for (int kt : kernels) {
        svm_parameter param = getDefaultParameter(C_SVC, kt);
        
        svm_model* model = svm_train(prob, &param);
        ASSERT_NE(model, nullptr);
        svm_free_and_destroy_model(&model);
    }
}

TEST_F(MemoryLeakTest, TrainWithDifferentSvmTypes) {
    {
        // Classification
        auto builder = createLinearlySeperableData(30, 42);
        svm_problem* prob = builder->build();
        
        for (int st : {C_SVC, NU_SVC}) {
            svm_parameter param = getDefaultParameter(st, RBF);
            svm_model* model = svm_train(prob, &param);
            ASSERT_NE(model, nullptr);
            svm_free_and_destroy_model(&model);
        }
    }
    
    {
        // Regression
        auto builder = createRegressionData(30, 0.1, 42);
        svm_problem* prob = builder->build();
        
        for (int st : {EPSILON_SVR, NU_SVR}) {
            svm_parameter param = getDefaultParameter(st, RBF);
            svm_model* model = svm_train(prob, &param);
            ASSERT_NE(model, nullptr);
            svm_free_and_destroy_model(&model);
        }
    }
    
    {
        // One-class
        auto builder = createLinearlySeperableData(30, 42);
        svm_problem* prob = builder->build();
        
        svm_parameter param = getDefaultParameter(ONE_CLASS, RBF);
        param.nu = 0.1;
        svm_model* model = svm_train(prob, &param);
        ASSERT_NE(model, nullptr);
        svm_free_and_destroy_model(&model);
    }
}

TEST_F(MemoryLeakTest, TrainWithProbability) {
    auto builder = createLinearlySeperableData(50, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.probability = 1;
    
    svm_model* model = svm_train(prob, &param);
    ASSERT_NE(model, nullptr);
    svm_free_and_destroy_model(&model);
}

TEST_F(MemoryLeakTest, MultiClassTrainAndFree) {
    auto builder = createMultiClassData(5, 30, 4, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    svm_model* model = svm_train(prob, &param);
    ASSERT_NE(model, nullptr);
    
    // Query model (should not leak)
    int nr_class = svm_get_nr_class(model);
    std::vector<int> labels(nr_class);
    svm_get_labels(model, labels.data());
    
    svm_free_and_destroy_model(&model);
}

// ===========================================================================
// Model Save/Load Memory Tests
// ===========================================================================

TEST_F(MemoryLeakTest, SaveLoadAndFreeModel) {
    auto builder = createLinearlySeperableData(30, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    svm_model* model = svm_train(prob, &param);
    ASSERT_NE(model, nullptr);
    
    std::string path = getTempFilePath();
    svm_save_model(path.c_str(), model);
    svm_free_and_destroy_model(&model);
    
    svm_model* loaded = svm_load_model(path.c_str());
    ASSERT_NE(loaded, nullptr);
    svm_free_and_destroy_model(&loaded);
    
    deleteTempFile(path);
}

TEST_F(MemoryLeakTest, RepeatSaveLoadAndFree) {
    std::string path = getTempFilePath();
    
    for (int iter = 0; iter < 5; ++iter) {
        auto builder = createLinearlySeperableData(20, 42 + iter);
        svm_problem* prob = builder->build();
        svm_parameter param = getDefaultParameter(C_SVC, RBF);
        
        svm_model* model = svm_train(prob, &param);
        ASSERT_NE(model, nullptr);
        
        svm_save_model(path.c_str(), model);
        svm_free_and_destroy_model(&model);
        
        svm_model* loaded = svm_load_model(path.c_str());
        ASSERT_NE(loaded, nullptr);
        svm_free_and_destroy_model(&loaded);
    }
    
    deleteTempFile(path);
}

// ===========================================================================
// Prediction Memory Tests
// ===========================================================================

TEST_F(MemoryLeakTest, RepeatedPredictions) {
    auto builder = createLinearlySeperableData(50, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    // Many predictions should not leak memory
    for (int iter = 0; iter < 100; ++iter) {
        for (int i = 0; i < prob->l; ++i) {
            svm_predict(model.get(), prob->x[i]);
        }
    }
}

TEST_F(MemoryLeakTest, RepeatedPredictValues) {
    auto builder = createLinearlySeperableData(50, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    double dec_value;
    for (int iter = 0; iter < 100; ++iter) {
        for (int i = 0; i < prob->l; ++i) {
            svm_predict_values(model.get(), prob->x[i], &dec_value);
        }
    }
}

TEST_F(MemoryLeakTest, RepeatedProbabilityPredictions) {
    auto builder = createLinearlySeperableData(50, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.probability = 1;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    std::vector<double> probs(2);
    for (int iter = 0; iter < 50; ++iter) {
        for (int i = 0; i < prob->l; ++i) {
            svm_predict_probability(model.get(), prob->x[i], probs.data());
        }
    }
}

// ===========================================================================
// Cross-Validation Memory Tests
// ===========================================================================

TEST_F(MemoryLeakTest, CrossValidation) {
    auto builder = createLinearlySeperableData(100, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    std::vector<double> target(prob->l);
    
    // Cross-validation should clean up internal models
    svm_cross_validation(prob, &param, 5, target.data());
}

TEST_F(MemoryLeakTest, RepeatedCrossValidation) {
    auto builder = createLinearlySeperableData(50, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    std::vector<double> target(prob->l);
    
    for (int iter = 0; iter < 5; ++iter) {
        svm_cross_validation(prob, &param, 5, target.data());
    }
}

// ===========================================================================
// Parameter Memory Tests
// ===========================================================================

TEST_F(MemoryLeakTest, DestroyParameterWithWeights) {
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    param.nr_weight = 2;
    param.weight_label = (int*)malloc(2 * sizeof(int));
    param.weight = (double*)malloc(2 * sizeof(double));
    
    param.weight_label[0] = 1;
    param.weight_label[1] = -1;
    param.weight[0] = 1.0;
    param.weight[1] = 2.0;
    
    // svm_destroy_param should free these
    svm_destroy_param(&param);
    
    // After destroy, pointers should be null-able
    // (svm_destroy_param frees the memory)
}

// ===========================================================================
// SvmProblemBuilder Memory Tests
// ===========================================================================

TEST_F(MemoryLeakTest, ProblemBuilderCreationDestruction) {
    for (int iter = 0; iter < 10; ++iter) {
        SvmProblemBuilder builder;
        
        for (int i = 0; i < 100; ++i) {
            builder.addDenseSample(1.0, {0.5, 0.5, 0.5, 0.5});
            builder.addDenseSample(-1.0, {-0.5, -0.5, -0.5, -0.5});
        }
        
        svm_problem* prob = builder.build();
        EXPECT_NE(prob, nullptr);
        // builder destructor should clean up
    }
}

TEST_F(MemoryLeakTest, ProblemBuilderClear) {
    SvmProblemBuilder builder;
    
    for (int iter = 0; iter < 10; ++iter) {
        for (int i = 0; i < 50; ++i) {
            builder.addDenseSample(1.0, {0.5, 0.5});
        }
        
        builder.build();
        builder.clear();
    }
}

// ===========================================================================
// Large Data Memory Tests
// ===========================================================================

TEST_F(MemoryLeakTest, LargeDatasetTrainAndFree) {
    auto builder = createLinearlySeperableData(500, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    svm_model* model = svm_train(prob, &param);
    ASSERT_NE(model, nullptr);
    
    // Make predictions
    for (int i = 0; i < prob->l; ++i) {
        svm_predict(model, prob->x[i]);
    }
    
    svm_free_and_destroy_model(&model);
}

TEST_F(MemoryLeakTest, HighDimensionalData) {
    SvmProblemBuilder builder;
    
    // High-dimensional sparse data
    for (int i = 0; i < 100; ++i) {
        std::vector<std::pair<int, double>> features;
        for (int j = 0; j < 10; ++j) {
            features.emplace_back(i * 100 + j, 1.0);
        }
        builder.addSample(i % 2 == 0 ? 1.0 : -1.0, features);
    }
    
    svm_problem* prob = builder.build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    svm_model* model = svm_train(prob, &param);
    ASSERT_NE(model, nullptr);
    svm_free_and_destroy_model(&model);
}

// ===========================================================================
// SvmModelGuard Tests
// ===========================================================================

TEST_F(MemoryLeakTest, SvmModelGuardBasic) {
    auto builder = createLinearlySeperableData(30, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    {
        SvmModelGuard guard(svm_train(prob, &param));
        ASSERT_TRUE(guard);
        svm_predict(guard.get(), prob->x[0]);
        // guard destructor should free model
    }
}

TEST_F(MemoryLeakTest, SvmModelGuardMove) {
    auto builder = createLinearlySeperableData(30, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    SvmModelGuard guard1(svm_train(prob, &param));
    ASSERT_TRUE(guard1);
    
    SvmModelGuard guard2 = std::move(guard1);
    EXPECT_FALSE(guard1);
    EXPECT_TRUE(guard2);
    
    // Only guard2 destructor should free
}

TEST_F(MemoryLeakTest, SvmModelGuardRelease) {
    auto builder = createLinearlySeperableData(30, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    svm_model* raw_model;
    {
        SvmModelGuard guard(svm_train(prob, &param));
        ASSERT_TRUE(guard);
        raw_model = guard.release();
        EXPECT_FALSE(guard);
    }
    
    // Manual cleanup needed
    svm_free_and_destroy_model(&raw_model);
}

TEST_F(MemoryLeakTest, SvmModelGuardReset) {
    auto builder = createLinearlySeperableData(30, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    SvmModelGuard guard(svm_train(prob, &param));
    ASSERT_TRUE(guard);
    
    // Reset with new model
    guard.reset(svm_train(prob, &param));
    EXPECT_TRUE(guard);
    
    // Reset to null
    guard.reset();
    EXPECT_FALSE(guard);
}
