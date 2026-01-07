/**
 * @file test_cross_validation.cpp
 * @brief Integration tests for SVM cross-validation
 */

#include <gtest/gtest.h>
#include "svm.h"
#include "test_utils.h"
#include <vector>
#include <numeric>
#include <cmath>

using namespace libsvm_test;

class CrossValidationTest : public ::testing::Test {
protected:
    void SetUp() override {
        suppressOutput();
    }
    
    void TearDown() override {
        restoreOutput();
    }
};

// ===========================================================================
// Basic Cross-Validation Tests
// ===========================================================================

TEST_F(CrossValidationTest, FiveFoldCV_BinaryClassification) {
    auto builder = createLinearlySeperableData(100, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.5;
    
    std::vector<double> target(prob->l);
    
    svm_cross_validation(prob, &param, 5, target.data());
    
    // Calculate CV accuracy
    int correct = 0;
    for (int i = 0; i < prob->l; ++i) {
        if (target[i] == prob->y[i]) ++correct;
    }
    
    double accuracy = static_cast<double>(correct) / prob->l;
    EXPECT_GT(accuracy, 0.85) << "5-fold CV accuracy: " << accuracy * 100 << "%";
}

TEST_F(CrossValidationTest, TenFoldCV_BinaryClassification) {
    auto builder = createLinearlySeperableData(100, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.5;
    
    std::vector<double> target(prob->l);
    
    svm_cross_validation(prob, &param, 10, target.data());
    
    int correct = 0;
    for (int i = 0; i < prob->l; ++i) {
        if (target[i] == prob->y[i]) ++correct;
    }
    
    double accuracy = static_cast<double>(correct) / prob->l;
    EXPECT_GT(accuracy, 0.85);
}

TEST_F(CrossValidationTest, LeaveOneOutCV) {
    auto builder = createLinearlySeperableData(20, 42);  // Small dataset for LOO
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.5;
    
    std::vector<double> target(prob->l);
    
    // n-fold CV is effectively leave-one-out
    svm_cross_validation(prob, &param, prob->l, target.data());
    
    int correct = 0;
    for (int i = 0; i < prob->l; ++i) {
        if (target[i] == prob->y[i]) ++correct;
    }
    
    double accuracy = static_cast<double>(correct) / prob->l;
    EXPECT_GT(accuracy, 0.7);  // LOO might have lower accuracy
}

TEST_F(CrossValidationTest, TwoFoldCV) {
    auto builder = createLinearlySeperableData(100, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.5;
    
    std::vector<double> target(prob->l);
    
    svm_cross_validation(prob, &param, 2, target.data());
    
    int correct = 0;
    for (int i = 0; i < prob->l; ++i) {
        if (target[i] == prob->y[i]) ++correct;
    }
    
    double accuracy = static_cast<double>(correct) / prob->l;
    EXPECT_GT(accuracy, 0.75);
}

// ===========================================================================
// Multi-class Cross-Validation Tests
// ===========================================================================

TEST_F(CrossValidationTest, FiveFoldCV_MultiClass) {
    auto builder = createMultiClassData(4, 50, 4, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.5;
    
    std::vector<double> target(prob->l);
    
    svm_cross_validation(prob, &param, 5, target.data());
    
    int correct = 0;
    for (int i = 0; i < prob->l; ++i) {
        if (target[i] == prob->y[i]) ++correct;
    }
    
    double accuracy = static_cast<double>(correct) / prob->l;
    EXPECT_GT(accuracy, 0.7);
}

// ===========================================================================
// Regression Cross-Validation Tests
// ===========================================================================

TEST_F(CrossValidationTest, FiveFoldCV_EpsilonSVR) {
    auto builder = createRegressionData(100, 0.1, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(EPSILON_SVR, RBF);
    param.gamma = 0.5;
    param.p = 0.1;
    param.C = 10;
    
    std::vector<double> target(prob->l);
    
    svm_cross_validation(prob, &param, 5, target.data());
    
    // Calculate MSE
    double mse = 0;
    for (int i = 0; i < prob->l; ++i) {
        double diff = target[i] - prob->y[i];
        mse += diff * diff;
    }
    mse /= prob->l;
    
    EXPECT_LT(mse, 5.0) << "CV MSE: " << mse;
}

TEST_F(CrossValidationTest, FiveFoldCV_NuSVR) {
    auto builder = createRegressionData(100, 0.1, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(NU_SVR, RBF);
    param.gamma = 0.5;
    param.nu = 0.5;
    param.C = 10;
    
    std::vector<double> target(prob->l);
    
    svm_cross_validation(prob, &param, 5, target.data());
    
    double mse = 0;
    for (int i = 0; i < prob->l; ++i) {
        double diff = target[i] - prob->y[i];
        mse += diff * diff;
    }
    mse /= prob->l;
    
    EXPECT_LT(mse, 10.0);
}

// ===========================================================================
// Different Kernel Cross-Validation Tests
// ===========================================================================

TEST_F(CrossValidationTest, CV_LinearKernel) {
    auto builder = createLinearlySeperableData(100, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, LINEAR);
    
    std::vector<double> target(prob->l);
    
    svm_cross_validation(prob, &param, 5, target.data());
    
    int correct = 0;
    for (int i = 0; i < prob->l; ++i) {
        if (target[i] == prob->y[i]) ++correct;
    }
    
    double accuracy = static_cast<double>(correct) / prob->l;
    EXPECT_GT(accuracy, 0.9);
}

TEST_F(CrossValidationTest, CV_PolynomialKernel) {
    auto builder = createXorData(30, 0.05, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, POLY);
    param.degree = 2;
    param.gamma = 1.0;
    param.coef0 = 1.0;
    param.C = 10;
    
    std::vector<double> target(prob->l);
    
    svm_cross_validation(prob, &param, 5, target.data());
    
    int correct = 0;
    for (int i = 0; i < prob->l; ++i) {
        if (target[i] == prob->y[i]) ++correct;
    }
    
    double accuracy = static_cast<double>(correct) / prob->l;
    EXPECT_GT(accuracy, 0.7);
}

// ===========================================================================
// Stability Tests
// ===========================================================================

TEST_F(CrossValidationTest, CV_Reproducibility) {
    auto builder = createLinearlySeperableData(100, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.5;
    
    std::vector<double> target1(prob->l);
    std::vector<double> target2(prob->l);
    
    svm_cross_validation(prob, &param, 5, target1.data());
    svm_cross_validation(prob, &param, 5, target2.data());
    
    // Results should be the same for same data and parameters
    // Note: This may fail if there's randomness in the implementation
    for (int i = 0; i < prob->l; ++i) {
        EXPECT_DOUBLE_EQ(target1[i], target2[i]);
    }
}

TEST_F(CrossValidationTest, CV_VaryingFolds) {
    auto builder = createLinearlySeperableData(100, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.5;
    
    std::vector<int> folds = {2, 3, 5, 10, 20};
    std::vector<double> accuracies;
    
    for (int k : folds) {
        std::vector<double> target(prob->l);
        svm_cross_validation(prob, &param, k, target.data());
        
        int correct = 0;
        for (int i = 0; i < prob->l; ++i) {
            if (target[i] == prob->y[i]) ++correct;
        }
        
        accuracies.push_back(static_cast<double>(correct) / prob->l);
    }
    
    // All CV runs should give reasonable accuracy
    for (size_t i = 0; i < accuracies.size(); ++i) {
        EXPECT_GT(accuracies[i], 0.7) << "k=" << folds[i];
    }
}

// ===========================================================================
// Heart Scale Dataset Cross-Validation
// ===========================================================================

TEST_F(CrossValidationTest, HeartScale_FiveFoldCV) {
    std::string filepath = std::string(TEST_DATA_DIR) + "/heart_scale";
    auto builder = loadHeartScale(filepath);
    
    if (builder->size() == 0) {
        GTEST_SKIP() << "heart_scale file not found";
    }
    
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.03125;
    param.C = 8.0;
    
    std::vector<double> target(prob->l);
    
    svm_cross_validation(prob, &param, 5, target.data());
    
    int correct = 0;
    for (int i = 0; i < prob->l; ++i) {
        if (target[i] == prob->y[i]) ++correct;
    }
    
    double accuracy = static_cast<double>(correct) / prob->l;
    EXPECT_GT(accuracy, 0.8) << "Heart scale 5-fold CV accuracy: " << accuracy * 100 << "%";
}

// ===========================================================================
// Edge Cases
// ===========================================================================

TEST_F(CrossValidationTest, CV_SmallDataset) {
    auto builder = std::make_unique<SvmProblemBuilder>();
    
    for (int i = 0; i < 5; ++i) {
        builder->addDenseSample(1.0, {1.0 + i * 0.1, 1.0});
        builder->addDenseSample(-1.0, {-1.0 - i * 0.1, -1.0});
    }
    
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, LINEAR);
    
    std::vector<double> target(prob->l);
    
    svm_cross_validation(prob, &param, 5, target.data());
    
    // Should complete without error
    SUCCEED();
}

TEST_F(CrossValidationTest, CV_ImbalancedClasses) {
    auto builder = std::make_unique<SvmProblemBuilder>();
    
    // 80 positive, 20 negative
    for (int i = 0; i < 80; ++i) {
        builder->addDenseSample(1.0, {1.0 + i * 0.01, 1.0 + i * 0.01});
    }
    for (int i = 0; i < 20; ++i) {
        builder->addDenseSample(-1.0, {-1.0 - i * 0.01, -1.0 - i * 0.01});
    }
    
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.5;
    
    std::vector<double> target(prob->l);
    
    svm_cross_validation(prob, &param, 5, target.data());
    
    int correct = 0;
    for (int i = 0; i < prob->l; ++i) {
        if (target[i] == prob->y[i]) ++correct;
    }
    
    double accuracy = static_cast<double>(correct) / prob->l;
    EXPECT_GT(accuracy, 0.7);
}
