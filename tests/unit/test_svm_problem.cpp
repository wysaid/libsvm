/**
 * @file test_svm_problem.cpp
 * @brief Unit tests for svm_problem structure
 */

#include <gtest/gtest.h>
#include "svm.h"
#include "test_utils.h"
#include <vector>

using namespace libsvm_test;

class SvmProblemTest : public ::testing::Test {
protected:
    void SetUp() override {
        suppressOutput();
    }
    
    void TearDown() override {
        restoreOutput();
    }
};

// Test SvmProblemBuilder basic functionality
TEST_F(SvmProblemTest, BuilderBasicUsage) {
    SvmProblemBuilder builder;
    
    builder.addDenseSample(1.0, {0.5, 0.5});
    builder.addDenseSample(-1.0, {-0.5, -0.5});
    
    EXPECT_EQ(builder.size(), 2);
    
    svm_problem* prob = builder.build();
    ASSERT_NE(prob, nullptr);
    EXPECT_EQ(prob->l, 2);
    EXPECT_DOUBLE_EQ(prob->y[0], 1.0);
    EXPECT_DOUBLE_EQ(prob->y[1], -1.0);
}

// Test SvmProblemBuilder with sparse samples
TEST_F(SvmProblemTest, BuilderSparseSamples) {
    SvmProblemBuilder builder;
    
    builder.addSample(1.0, {{1, 0.5}, {3, 1.0}, {10, -0.5}});
    builder.addSample(-1.0, {{2, 0.3}, {5, 0.8}});
    
    svm_problem* prob = builder.build();
    ASSERT_NE(prob, nullptr);
    EXPECT_EQ(prob->l, 2);
    
    // Check first sample features
    EXPECT_EQ(prob->x[0][0].index, 1);
    EXPECT_DOUBLE_EQ(prob->x[0][0].value, 0.5);
    EXPECT_EQ(prob->x[0][1].index, 3);
    EXPECT_DOUBLE_EQ(prob->x[0][1].value, 1.0);
    EXPECT_EQ(prob->x[0][2].index, 10);
    EXPECT_DOUBLE_EQ(prob->x[0][2].value, -0.5);
    EXPECT_EQ(prob->x[0][3].index, -1);  // terminator
}

// Test SvmProblemBuilder clear
TEST_F(SvmProblemTest, BuilderClear) {
    SvmProblemBuilder builder;
    
    builder.addDenseSample(1.0, {0.5, 0.5});
    EXPECT_EQ(builder.size(), 1);
    
    builder.clear();
    EXPECT_EQ(builder.size(), 0);
    
    builder.addDenseSample(-1.0, {-0.5, -0.5});
    EXPECT_EQ(builder.size(), 1);
}

// Test svm_problem structure layout
TEST_F(SvmProblemTest, StructureLayout) {
    auto builder = createLinearlySeperableData(20);
    svm_problem* prob = builder->build();
    
    ASSERT_NE(prob, nullptr);
    ASSERT_NE(prob->y, nullptr);
    ASSERT_NE(prob->x, nullptr);
    EXPECT_EQ(prob->l, 40);  // 20 samples per class
    
    // Verify each sample has valid data
    for (int i = 0; i < prob->l; ++i) {
        ASSERT_NE(prob->x[i], nullptr);
        
        // Find terminator
        int j = 0;
        while (prob->x[i][j].index != -1) {
            EXPECT_GT(prob->x[i][j].index, 0);
            ++j;
            EXPECT_LT(j, 1000);  // Sanity check
        }
    }
}

// Test linearly separable data generation
TEST_F(SvmProblemTest, LinearlySeperableData) {
    auto builder = createLinearlySeperableData(50, 42);
    svm_problem* prob = builder->build();
    
    EXPECT_EQ(prob->l, 100);
    
    int positive = 0, negative = 0;
    for (int i = 0; i < prob->l; ++i) {
        if (prob->y[i] > 0) ++positive;
        else if (prob->y[i] < 0) ++negative;
    }
    
    EXPECT_EQ(positive, 50);
    EXPECT_EQ(negative, 50);
}

// Test XOR data generation
TEST_F(SvmProblemTest, XorData) {
    auto builder = createXorData(10, 0.1, 42);
    svm_problem* prob = builder->build();
    
    EXPECT_EQ(prob->l, 40);  // 10 samples per quadrant * 4 quadrants
    
    int positive = 0, negative = 0;
    for (int i = 0; i < prob->l; ++i) {
        if (prob->y[i] > 0) ++positive;
        else if (prob->y[i] < 0) ++negative;
    }
    
    EXPECT_EQ(positive, 20);
    EXPECT_EQ(negative, 20);
}

// Test multi-class data generation
TEST_F(SvmProblemTest, MultiClassData) {
    auto builder = createMultiClassData(4, 25, 5, 42);
    svm_problem* prob = builder->build();
    
    EXPECT_EQ(prob->l, 100);  // 4 classes * 25 samples
    
    // Count samples per class
    std::map<double, int> class_counts;
    for (int i = 0; i < prob->l; ++i) {
        class_counts[prob->y[i]]++;
    }
    
    EXPECT_EQ(class_counts.size(), 4);
    for (const auto& [label, count] : class_counts) {
        EXPECT_EQ(count, 25);
    }
}

// Test regression data generation
TEST_F(SvmProblemTest, RegressionData) {
    auto builder = createRegressionData(100, 0.1, 42);
    svm_problem* prob = builder->build();
    
    EXPECT_EQ(prob->l, 100);
    
    // Labels should be continuous (not just +1/-1)
    bool has_varied_labels = false;
    double first_label = prob->y[0];
    for (int i = 1; i < prob->l; ++i) {
        if (std::abs(prob->y[i] - first_label) > 0.5) {
            has_varied_labels = true;
            break;
        }
    }
    EXPECT_TRUE(has_varied_labels);
}

// Test loading heart_scale data
TEST_F(SvmProblemTest, LoadHeartScale) {
    std::string filepath = std::string(TEST_DATA_DIR) + "/heart_scale";
    auto builder = loadHeartScale(filepath);
    
    if (builder->size() == 0) {
        GTEST_SKIP() << "heart_scale file not found, skipping test";
    }
    
    svm_problem* prob = builder->build();
    
    EXPECT_EQ(prob->l, 270);  // heart_scale has 270 samples
    
    // Check labels are binary
    for (int i = 0; i < prob->l; ++i) {
        EXPECT_TRUE(prob->y[i] == 1.0 || prob->y[i] == -1.0);
    }
}

// Test empty problem
TEST_F(SvmProblemTest, EmptyProblem) {
    SvmProblemBuilder builder;
    
    svm_problem* prob = builder.build();
    EXPECT_EQ(prob, nullptr);
}

// Test single sample problem
TEST_F(SvmProblemTest, SingleSample) {
    SvmProblemBuilder builder;
    builder.addDenseSample(1.0, {0.5, 0.5, 0.5});
    
    svm_problem* prob = builder.build();
    ASSERT_NE(prob, nullptr);
    EXPECT_EQ(prob->l, 1);
    EXPECT_DOUBLE_EQ(prob->y[0], 1.0);
}

// Test high-dimensional data
TEST_F(SvmProblemTest, HighDimensionalData) {
    SvmProblemBuilder builder;
    
    // Create a 1000-dimensional sample
    std::vector<double> features(1000);
    for (int i = 0; i < 1000; ++i) {
        features[i] = static_cast<double>(i) / 1000.0;
    }
    
    builder.addDenseSample(1.0, features);
    
    svm_problem* prob = builder.build();
    ASSERT_NE(prob, nullptr);
    EXPECT_EQ(prob->l, 1);
    
    // Count features (non-zero)
    int count = 0;
    while (prob->x[0][count].index != -1) {
        ++count;
    }
    EXPECT_EQ(count, 999);  // First element is 0, so excluded
}

// Test data consistency after rebuild
TEST_F(SvmProblemTest, RebuildConsistency) {
    SvmProblemBuilder builder;
    builder.addDenseSample(1.0, {0.5, 0.5});
    builder.addDenseSample(-1.0, {-0.5, -0.5});
    
    svm_problem* prob1 = builder.build();
    svm_problem* prob2 = builder.build();
    
    EXPECT_EQ(prob1, prob2);  // Should return same pointer
    EXPECT_EQ(prob1->l, prob2->l);
}

// Test deterministic data generation with same seed
TEST_F(SvmProblemTest, DeterministicGeneration) {
    auto builder1 = createLinearlySeperableData(10, 42);
    auto builder2 = createLinearlySeperableData(10, 42);
    
    svm_problem* prob1 = builder1->build();
    svm_problem* prob2 = builder2->build();
    
    ASSERT_EQ(prob1->l, prob2->l);
    
    for (int i = 0; i < prob1->l; ++i) {
        EXPECT_DOUBLE_EQ(prob1->y[i], prob2->y[i]);
        
        int j = 0;
        while (prob1->x[i][j].index != -1) {
            EXPECT_EQ(prob1->x[i][j].index, prob2->x[i][j].index);
            EXPECT_DOUBLE_EQ(prob1->x[i][j].value, prob2->x[i][j].value);
            ++j;
        }
    }
}

// Test different seeds produce different data
TEST_F(SvmProblemTest, DifferentSeeds) {
    auto builder1 = createLinearlySeperableData(10, 42);
    auto builder2 = createLinearlySeperableData(10, 123);
    
    svm_problem* prob1 = builder1->build();
    svm_problem* prob2 = builder2->build();
    
    bool has_difference = false;
    for (int i = 0; i < prob1->l; ++i) {
        int j = 0;
        while (prob1->x[i][j].index != -1) {
            if (prob1->x[i][j].value != prob2->x[i][j].value) {
                has_difference = true;
                break;
            }
            ++j;
        }
        if (has_difference) break;
    }
    
    EXPECT_TRUE(has_difference);
}
