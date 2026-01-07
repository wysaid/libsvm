/**
 * @file test_resource_management.cpp
 * @brief Tests for proper resource management and RAII patterns
 */

#include <gtest/gtest.h>
#include "svm.h"
#include "test_utils.h"
#include <vector>
#include <thread>
#include <future>

using namespace libsvm_test;

class ResourceManagementTest : public ::testing::Test {
protected:
    void SetUp() override {
        suppressOutput();
    }
    
    void TearDown() override {
        restoreOutput();
    }
};

// ===========================================================================
// Exception Safety Tests
// ===========================================================================

TEST_F(ResourceManagementTest, SvmModelGuardExceptionSafety) {
    auto builder = createLinearlySeperableData(30, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    try {
        SvmModelGuard guard(svm_train(prob, &param));
        ASSERT_TRUE(guard);
        
        // Simulate work that might throw
        throw std::runtime_error("Test exception");
        
        // Never reached
        FAIL();
    } catch (const std::runtime_error&) {
        // Guard should have freed the model
        SUCCEED();
    }
}

TEST_F(ResourceManagementTest, SvmProblemBuilderExceptionSafety) {
    try {
        SvmProblemBuilder builder;
        
        for (int i = 0; i < 100; ++i) {
            builder.addDenseSample(1.0, {0.5, 0.5});
        }
        
        throw std::runtime_error("Test exception");
        
        FAIL();
    } catch (const std::runtime_error&) {
        // Builder should have cleaned up
        SUCCEED();
    }
}

// ===========================================================================
// Concurrent Access Tests (Thread Safety)
// ===========================================================================

TEST_F(ResourceManagementTest, ConcurrentPredictions) {
    auto builder = createLinearlySeperableData(100, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    // Copy problem data for thread safety
    std::vector<std::vector<svm_node>> test_samples;
    for (int i = 0; i < prob->l; ++i) {
        std::vector<svm_node> sample;
        int j = 0;
        while (prob->x[i][j].index != -1) {
            sample.push_back(prob->x[i][j]);
            ++j;
        }
        sample.push_back({-1, 0});
        test_samples.push_back(sample);
    }
    
    // Concurrent predictions from multiple threads
    auto predict_fn = [&model, &test_samples](int start, int end) {
        std::vector<double> results;
        for (int i = start; i < end; ++i) {
            double pred = svm_predict(model.get(), test_samples[i].data());
            results.push_back(pred);
        }
        return results;
    };
    
    int n = static_cast<int>(test_samples.size());
    auto future1 = std::async(std::launch::async, predict_fn, 0, n / 2);
    auto future2 = std::async(std::launch::async, predict_fn, n / 2, n);
    
    auto results1 = future1.get();
    auto results2 = future2.get();
    
    EXPECT_EQ(results1.size() + results2.size(), static_cast<size_t>(n));
}

TEST_F(ResourceManagementTest, ConcurrentModelTraining) {
    // Train multiple models concurrently
    auto train_fn = [](int seed) {
        auto builder = createLinearlySeperableData(30, seed);
        svm_problem* prob = builder->build();
        svm_parameter param = getDefaultParameter(C_SVC, RBF);
        
        svm_model* model = svm_train(prob, &param);
        if (model) {
            int nr_sv = svm_get_nr_sv(model);
            svm_free_and_destroy_model(&model);
            return nr_sv;
        }
        return -1;
    };
    
    auto future1 = std::async(std::launch::async, train_fn, 42);
    auto future2 = std::async(std::launch::async, train_fn, 123);
    auto future3 = std::async(std::launch::async, train_fn, 456);
    
    int sv1 = future1.get();
    int sv2 = future2.get();
    int sv3 = future3.get();
    
    EXPECT_GT(sv1, 0);
    EXPECT_GT(sv2, 0);
    EXPECT_GT(sv3, 0);
}

// ===========================================================================
// Resource Cleanup Order Tests
// ===========================================================================

TEST_F(ResourceManagementTest, CleanupOrderModelFirst) {
    auto builder = createLinearlySeperableData(30, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    svm_model* model = svm_train(prob, &param);
    ASSERT_NE(model, nullptr);
    
    // Free model first, then problem builder goes out of scope
    svm_free_and_destroy_model(&model);
}

TEST_F(ResourceManagementTest, CleanupOrderBuilderFirst) {
    svm_model* model = nullptr;
    
    {
        auto builder = createLinearlySeperableData(30, 42);
        svm_problem* prob = builder->build();
        svm_parameter param = getDefaultParameter(C_SVC, RBF);
        
        model = svm_train(prob, &param);
        // Builder goes out of scope, but model is saved
    }
    
    // Model should still be valid (it copies the data)
    ASSERT_NE(model, nullptr);
    
    // Create a test sample for prediction
    std::vector<svm_node> test = {{1, 1.0}, {2, 1.0}, {-1, 0}};
    double pred = svm_predict(model, test.data());
    EXPECT_TRUE(pred == 1.0 || pred == -1.0);
    
    svm_free_and_destroy_model(&model);
}

// ===========================================================================
// Double-Free Protection Tests
// ===========================================================================

TEST_F(ResourceManagementTest, DoubleDestroyProtection) {
    auto builder = createLinearlySeperableData(20, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    svm_model* model = svm_train(prob, &param);
    ASSERT_NE(model, nullptr);
    
    svm_free_and_destroy_model(&model);
    EXPECT_EQ(model, nullptr);
    
    // Second destroy should be safe (pointer is null)
    svm_free_and_destroy_model(&model);
    EXPECT_EQ(model, nullptr);
}

TEST_F(ResourceManagementTest, SvmModelGuardDoubleReset) {
    auto builder = createLinearlySeperableData(20, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    SvmModelGuard guard(svm_train(prob, &param));
    
    guard.reset();
    EXPECT_FALSE(guard);
    
    guard.reset();  // Second reset should be safe
    EXPECT_FALSE(guard);
}

// ===========================================================================
// Null Pointer Safety Tests
// ===========================================================================

TEST_F(ResourceManagementTest, NullModelOperations) {
    SvmModelGuard guard(nullptr);
    EXPECT_FALSE(guard);
    EXPECT_EQ(guard.get(), nullptr);
}

TEST_F(ResourceManagementTest, SvmFreeNullModel) {
    svm_model* null_model = nullptr;
    
    // Should not crash
    svm_free_and_destroy_model(&null_model);
    EXPECT_EQ(null_model, nullptr);
}

// ===========================================================================
// File Handle Tests
// ===========================================================================

TEST_F(ResourceManagementTest, FileHandleCleanup) {
    auto builder = createLinearlySeperableData(20, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    SvmModelGuard model(svm_train(prob, &param));
    
    std::string path = getTempFilePath();
    
    // Save and load multiple times
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(svm_save_model(path.c_str(), model.get()), 0);
        
        SvmModelGuard loaded(svm_load_model(path.c_str()));
        EXPECT_TRUE(loaded);
    }
    
    deleteTempFile(path);
}

// ===========================================================================
// Large-Scale Resource Tests
// ===========================================================================

TEST_F(ResourceManagementTest, ManyModelsSequential) {
    for (int i = 0; i < 20; ++i) {
        auto builder = createLinearlySeperableData(20, 42 + i);
        svm_problem* prob = builder->build();
        svm_parameter param = getDefaultParameter(C_SVC, RBF);
        
        SvmModelGuard model(svm_train(prob, &param));
        EXPECT_TRUE(model);
        
        // Do some predictions
        for (int j = 0; j < prob->l; ++j) {
            svm_predict(model.get(), prob->x[j]);
        }
    }
}

TEST_F(ResourceManagementTest, ManyBuildersSequential) {
    for (int i = 0; i < 50; ++i) {
        auto builder = createLinearlySeperableData(50, 42 + i);
        svm_problem* prob = builder->build();
        EXPECT_NE(prob, nullptr);
        EXPECT_EQ(prob->l, 100);
    }
}

// ===========================================================================
// Stress Tests
// ===========================================================================

TEST_F(ResourceManagementTest, RapidAllocationDeallocation) {
    for (int i = 0; i < 100; ++i) {
        SvmProblemBuilder builder;
        builder.addDenseSample(1.0, {1.0, 2.0, 3.0});
        builder.addDenseSample(-1.0, {-1.0, -2.0, -3.0});
        builder.build();
        builder.clear();
    }
}

TEST_F(ResourceManagementTest, AlternatingOperations) {
    std::vector<std::string> temp_files;
    
    for (int i = 0; i < 10; ++i) {
        // Create problem
        auto builder = createLinearlySeperableData(20, 42 + i);
        svm_problem* prob = builder->build();
        svm_parameter param = getDefaultParameter(C_SVC, RBF);
        
        // Train
        SvmModelGuard model(svm_train(prob, &param));
        
        // Save
        std::string path = getTempFilePath();
        temp_files.push_back(path);
        svm_save_model(path.c_str(), model.get());
        
        // Load and predict
        SvmModelGuard loaded(svm_load_model(path.c_str()));
        svm_predict(loaded.get(), prob->x[0]);
        
        // Cross validation
        std::vector<double> target(prob->l);
        svm_cross_validation(prob, &param, 2, target.data());
    }
    
    // Cleanup
    for (const auto& path : temp_files) {
        deleteTempFile(path);
    }
}
