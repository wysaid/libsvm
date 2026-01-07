/**
 * @file test_train_predict.cpp
 * @brief Integration tests for SVM training and prediction workflow
 */

#include <gtest/gtest.h>
#include "svm.h"
#include "test_utils.h"
#include <vector>
#include <cmath>

using namespace libsvm_test;

class TrainPredictTest : public ::testing::Test {
protected:
    void SetUp() override {
        suppressOutput();
    }
    
    void TearDown() override {
        restoreOutput();
    }
};

// ===========================================================================
// Basic Training and Prediction Tests
// ===========================================================================

TEST_F(TrainPredictTest, BasicBinaryClassification) {
    auto builder = createLinearlySeperableData(50, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    // Test predictions on training data (should have high accuracy)
    int correct = 0;
    for (int i = 0; i < prob->l; ++i) {
        double pred = svm_predict(model.get(), prob->x[i]);
        if (pred == prob->y[i]) ++correct;
    }
    
    double accuracy = static_cast<double>(correct) / prob->l;
    EXPECT_GT(accuracy, 0.9) << "Training accuracy: " << accuracy * 100 << "%";
}

TEST_F(TrainPredictTest, LinearKernelClassification) {
    auto builder = createLinearlySeperableData(50, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, LINEAR);
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    // Linear kernel should work well on linearly separable data
    int correct = 0;
    for (int i = 0; i < prob->l; ++i) {
        double pred = svm_predict(model.get(), prob->x[i]);
        if (pred == prob->y[i]) ++correct;
    }
    
    double accuracy = static_cast<double>(correct) / prob->l;
    EXPECT_GT(accuracy, 0.95);
}

TEST_F(TrainPredictTest, RBFKernelXorData) {
    auto builder = createXorData(20, 0.05, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 1.0;
    param.C = 10;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    int correct = 0;
    for (int i = 0; i < prob->l; ++i) {
        double pred = svm_predict(model.get(), prob->x[i]);
        if (pred == prob->y[i]) ++correct;
    }
    
    double accuracy = static_cast<double>(correct) / prob->l;
    EXPECT_GT(accuracy, 0.85) << "XOR accuracy: " << accuracy * 100 << "%";
}

TEST_F(TrainPredictTest, PolynomialKernelXorData) {
    auto builder = createXorData(20, 0.05, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, POLY);
    param.degree = 2;
    param.gamma = 1.0;
    param.coef0 = 1.0;
    param.C = 10;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    int correct = 0;
    for (int i = 0; i < prob->l; ++i) {
        double pred = svm_predict(model.get(), prob->x[i]);
        if (pred == prob->y[i]) ++correct;
    }
    
    double accuracy = static_cast<double>(correct) / prob->l;
    EXPECT_GT(accuracy, 0.80);
}

// ===========================================================================
// Multi-class Classification Tests
// ===========================================================================

TEST_F(TrainPredictTest, MultiClassClassification) {
    auto builder = createMultiClassData(5, 30, 4, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.5;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    EXPECT_EQ(svm_get_nr_class(model.get()), 5);
    
    int correct = 0;
    for (int i = 0; i < prob->l; ++i) {
        double pred = svm_predict(model.get(), prob->x[i]);
        if (pred == prob->y[i]) ++correct;
    }
    
    double accuracy = static_cast<double>(correct) / prob->l;
    EXPECT_GT(accuracy, 0.8);
}

TEST_F(TrainPredictTest, MultiClassPredictValues) {
    auto builder = createMultiClassData(3, 30, 4, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    int nr_class = svm_get_nr_class(model.get());
    int nr_pairs = nr_class * (nr_class - 1) / 2;
    std::vector<double> dec_values(nr_pairs);
    
    // Predict with decision values
    double pred = svm_predict_values(model.get(), prob->x[0], dec_values.data());
    
    // Verify we got valid decision values
    for (double dv : dec_values) {
        EXPECT_FALSE(std::isnan(dv));
        EXPECT_FALSE(std::isinf(dv));
    }
}

// ===========================================================================
// Regression Tests
// ===========================================================================

TEST_F(TrainPredictTest, EpsilonSVR_BasicRegression) {
    auto builder = createRegressionData(100, 0.1, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(EPSILON_SVR, RBF);
    param.gamma = 0.5;
    param.p = 0.1;
    param.C = 10;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    // Calculate MSE on training data
    double mse = 0;
    for (int i = 0; i < prob->l; ++i) {
        double pred = svm_predict(model.get(), prob->x[i]);
        double diff = pred - prob->y[i];
        mse += diff * diff;
    }
    mse /= prob->l;
    
    EXPECT_LT(mse, 1.0) << "MSE: " << mse;
}

TEST_F(TrainPredictTest, NuSVR_BasicRegression) {
    auto builder = createRegressionData(100, 0.1, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(NU_SVR, RBF);
    param.gamma = 0.5;
    param.nu = 0.5;
    param.C = 10;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    double mse = 0;
    for (int i = 0; i < prob->l; ++i) {
        double pred = svm_predict(model.get(), prob->x[i]);
        double diff = pred - prob->y[i];
        mse += diff * diff;
    }
    mse /= prob->l;
    
    EXPECT_LT(mse, 2.0);
}

TEST_F(TrainPredictTest, SVR_LinearKernel) {
    auto builder = createRegressionData(100, 0.1, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(EPSILON_SVR, LINEAR);
    param.p = 0.1;
    param.C = 10;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    double mse = 0;
    for (int i = 0; i < prob->l; ++i) {
        double pred = svm_predict(model.get(), prob->x[i]);
        double diff = pred - prob->y[i];
        mse += diff * diff;
    }
    mse /= prob->l;
    
    EXPECT_LT(mse, 1.0);
}

// ===========================================================================
// One-Class SVM Tests
// ===========================================================================

TEST_F(TrainPredictTest, OneClass_NormalDataDetection) {
    auto builder = std::make_unique<SvmProblemBuilder>();
    
    // Normal data cluster
    for (int i = 0; i < 100; ++i) {
        builder->addDenseSample(1.0, {0.5 + (i % 10) * 0.05, 0.5 + (i / 10) * 0.05});
    }
    
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(ONE_CLASS, RBF);
    param.gamma = 2.0;
    param.nu = 0.1;  // Expect ~10% outliers
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    // Normal data should mostly be classified as +1
    int positives = 0;
    for (int i = 0; i < prob->l; ++i) {
        double pred = svm_predict(model.get(), prob->x[i]);
        if (pred > 0) ++positives;
    }
    
    double positive_rate = static_cast<double>(positives) / prob->l;
    EXPECT_GT(positive_rate, 0.85);
    
    // Outlier should be classified as -1
    std::vector<svm_node> outlier = {{1, 10.0}, {2, 10.0}, {-1, 0}};
    double outlier_pred = svm_predict(model.get(), outlier.data());
    EXPECT_EQ(outlier_pred, -1.0);
}

// ===========================================================================
// NU-SVC Tests
// ===========================================================================

TEST_F(TrainPredictTest, NuSVC_BasicClassification) {
    auto builder = createLinearlySeperableData(50, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(NU_SVC, RBF);
    param.nu = 0.5;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    int correct = 0;
    for (int i = 0; i < prob->l; ++i) {
        double pred = svm_predict(model.get(), prob->x[i]);
        if (pred == prob->y[i]) ++correct;
    }
    
    double accuracy = static_cast<double>(correct) / prob->l;
    EXPECT_GT(accuracy, 0.9);
}

// ===========================================================================
// Decision Values Tests
// ===========================================================================

TEST_F(TrainPredictTest, PredictValues_Binary) {
    auto builder = createLinearlySeperableData(50, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    double dec_value;
    double pred = svm_predict_values(model.get(), prob->x[0], &dec_value);
    
    // Decision value sign should match prediction
    if (pred > 0) {
        EXPECT_GT(dec_value, 0) << "Prediction: " << pred << ", Decision value: " << dec_value;
    } else {
        EXPECT_LT(dec_value, 0) << "Prediction: " << pred << ", Decision value: " << dec_value;
    }
}

// ===========================================================================
// Heart Scale Dataset Tests
// ===========================================================================

TEST_F(TrainPredictTest, HeartScale_Classification) {
    std::string filepath = std::string(TEST_DATA_DIR) + "/heart_scale";
    auto builder = loadHeartScale(filepath);
    
    if (builder->size() == 0) {
        GTEST_SKIP() << "heart_scale file not found";
    }
    
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.03125;  // Typical value for heart_scale
    param.C = 8.0;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    int correct = 0;
    for (int i = 0; i < prob->l; ++i) {
        double pred = svm_predict(model.get(), prob->x[i]);
        if (pred == prob->y[i]) ++correct;
    }
    
    double accuracy = static_cast<double>(correct) / prob->l;
    EXPECT_GT(accuracy, 0.85) << "Heart scale accuracy: " << accuracy * 100 << "%";
}

// ===========================================================================
// Edge Cases
// ===========================================================================

TEST_F(TrainPredictTest, SingleSamplePerClass) {
    auto builder = std::make_unique<SvmProblemBuilder>();
    
    builder->addDenseSample(1.0, {1.0, 1.0});
    builder->addDenseSample(-1.0, {-1.0, -1.0});
    
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, LINEAR);
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    // Should correctly classify training points
    std::vector<svm_node> pos = {{1, 1.0}, {2, 1.0}, {-1, 0}};
    std::vector<svm_node> neg = {{1, -1.0}, {2, -1.0}, {-1, 0}};
    
    EXPECT_DOUBLE_EQ(svm_predict(model.get(), pos.data()), 1.0);
    EXPECT_DOUBLE_EQ(svm_predict(model.get(), neg.data()), -1.0);
}

TEST_F(TrainPredictTest, IdenticalSamples) {
    auto builder = std::make_unique<SvmProblemBuilder>();
    
    // Same point with same label
    for (int i = 0; i < 10; ++i) {
        builder->addDenseSample(1.0, {1.0, 1.0});
    }
    for (int i = 0; i < 10; ++i) {
        builder->addDenseSample(-1.0, {-1.0, -1.0});
    }
    
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
}

TEST_F(TrainPredictTest, UnseenFeatures) {
    auto builder = createLinearlySeperableData(30, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    // Predict with features not seen in training (higher indices)
    std::vector<svm_node> test = {{1, 1.0}, {2, 1.0}, {100, 0.5}, {-1, 0}};
    double pred = svm_predict(model.get(), test.data());
    
    // Should not crash and return a valid prediction
    EXPECT_TRUE(pred == 1.0 || pred == -1.0);
}

TEST_F(TrainPredictTest, ImbalancedClasses) {
    auto builder = std::make_unique<SvmProblemBuilder>();
    
    // Imbalanced: 90 positive, 10 negative
    for (int i = 0; i < 90; ++i) {
        builder->addDenseSample(1.0, {1.0 + i * 0.01, 1.0 + i * 0.01});
    }
    for (int i = 0; i < 10; ++i) {
        builder->addDenseSample(-1.0, {-1.0 - i * 0.01, -1.0 - i * 0.01});
    }
    
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    // Should still classify minority class
    std::vector<svm_node> neg = {{1, -1.0}, {2, -1.0}, {-1, 0}};
    EXPECT_DOUBLE_EQ(svm_predict(model.get(), neg.data()), -1.0);
}

TEST_F(TrainPredictTest, ClassWeights) {
    auto builder = std::make_unique<SvmProblemBuilder>();
    
    // Imbalanced data
    for (int i = 0; i < 90; ++i) {
        builder->addDenseSample(1.0, {1.0 + i * 0.01, 1.0 + i * 0.01});
    }
    for (int i = 0; i < 10; ++i) {
        builder->addDenseSample(-1.0, {-1.0 - i * 0.01, -1.0 - i * 0.01});
    }
    
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    
    // Apply class weights to balance
    int weight_labels[] = {1, -1};
    double weights[] = {1.0, 9.0};  // Give more weight to minority class
    param.nr_weight = 2;
    param.weight_label = weight_labels;
    param.weight = weights;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    // Reset pointers before param goes out of scope
    param.weight_label = nullptr;
    param.weight = nullptr;
}
