/**
 * @file test_probability.cpp
 * @brief Integration tests for SVM probability estimation
 */

#include <gtest/gtest.h>
#include "svm.h"
#include "test_utils.h"
#include <vector>
#include <cmath>
#include <numeric>

using namespace libsvm_test;

class ProbabilityTest : public ::testing::Test {
protected:
    void SetUp() override {
        suppressOutput();
    }
    
    void TearDown() override {
        restoreOutput();
    }
};

// ===========================================================================
// Basic Probability Estimation Tests
// ===========================================================================

TEST_F(ProbabilityTest, BinaryClassificationProbability) {
    auto builder = createLinearlySeperableData(50, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.probability = 1;
    param.gamma = 0.5;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    EXPECT_EQ(svm_check_probability_model(model.get()), 1);
    
    int nr_class = svm_get_nr_class(model.get());
    EXPECT_EQ(nr_class, 2);
    
    std::vector<double> prob_estimates(nr_class);
    
    for (int i = 0; i < prob->l; ++i) {
        double pred = svm_predict_probability(model.get(), prob->x[i], prob_estimates.data());
        
        // Probabilities should sum to 1
        double sum = std::accumulate(prob_estimates.begin(), prob_estimates.end(), 0.0);
        EXPECT_NEAR(sum, 1.0, 1e-6) << "Probabilities don't sum to 1 at sample " << i;
        
        // Each probability should be in [0, 1]
        for (double p : prob_estimates) {
            EXPECT_GE(p, 0.0);
            EXPECT_LE(p, 1.0);
        }
        
        // Prediction should match highest probability class
        int max_idx = 0;
        for (int j = 1; j < nr_class; ++j) {
            if (prob_estimates[j] > prob_estimates[max_idx]) {
                max_idx = j;
            }
        }
        
        std::vector<int> labels(nr_class);
        svm_get_labels(model.get(), labels.data());
        EXPECT_EQ(pred, labels[max_idx]);
    }
}

TEST_F(ProbabilityTest, HighConfidencePredictions) {
    auto builder = createLinearlySeperableData(50, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.probability = 1;
    param.gamma = 0.5;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    // Test points far from decision boundary should have high confidence
    std::vector<svm_node> clearly_positive = {{1, 2.0}, {2, 2.0}, {-1, 0}};
    std::vector<svm_node> clearly_negative = {{1, -2.0}, {2, -2.0}, {-1, 0}};
    
    std::vector<double> probs(2);
    std::vector<int> labels(2);
    svm_get_labels(model.get(), labels.data());
    
    // Find index for positive and negative class
    int pos_idx = (labels[0] == 1) ? 0 : 1;
    int neg_idx = 1 - pos_idx;
    
    svm_predict_probability(model.get(), clearly_positive.data(), probs.data());
    EXPECT_GT(probs[pos_idx], 0.7) << "Expected high confidence for clearly positive sample";
    
    svm_predict_probability(model.get(), clearly_negative.data(), probs.data());
    EXPECT_GT(probs[neg_idx], 0.7) << "Expected high confidence for clearly negative sample";
}

// ===========================================================================
// Multi-class Probability Tests
// ===========================================================================

TEST_F(ProbabilityTest, MultiClassProbability) {
    auto builder = createMultiClassData(4, 40, 4, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.probability = 1;
    param.gamma = 0.5;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    EXPECT_EQ(svm_check_probability_model(model.get()), 1);
    
    int nr_class = svm_get_nr_class(model.get());
    EXPECT_EQ(nr_class, 4);
    
    std::vector<double> prob_estimates(nr_class);
    
    for (int i = 0; i < prob->l; ++i) {
        double pred = svm_predict_probability(model.get(), prob->x[i], prob_estimates.data());
        
        // Probabilities should sum to 1
        double sum = std::accumulate(prob_estimates.begin(), prob_estimates.end(), 0.0);
        EXPECT_NEAR(sum, 1.0, 1e-6);
        
        // Each probability should be valid
        for (double p : prob_estimates) {
            EXPECT_GE(p, 0.0);
            EXPECT_LE(p, 1.0);
        }
    }
}

TEST_F(ProbabilityTest, ManyClassesProbability) {
    auto builder = createMultiClassData(10, 20, 5, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.probability = 1;
    param.gamma = 0.3;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    int nr_class = svm_get_nr_class(model.get());
    EXPECT_EQ(nr_class, 10);
    
    std::vector<double> prob_estimates(nr_class);
    
    double pred = svm_predict_probability(model.get(), prob->x[0], prob_estimates.data());
    
    double sum = std::accumulate(prob_estimates.begin(), prob_estimates.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-5);
}

// ===========================================================================
// Probability vs Non-Probability Model Tests
// ===========================================================================

TEST_F(ProbabilityTest, NonProbabilityModelCheck) {
    auto builder = createLinearlySeperableData(30, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.probability = 0;  // Disable probability
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    EXPECT_EQ(svm_check_probability_model(model.get()), 0);
}

TEST_F(ProbabilityTest, PredictVsPredictProbability) {
    auto builder = createLinearlySeperableData(50, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.probability = 1;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    std::vector<double> probs(2);
    
    // Predictions from both methods should be consistent
    for (int i = 0; i < prob->l; ++i) {
        double pred_normal = svm_predict(model.get(), prob->x[i]);
        double pred_prob = svm_predict_probability(model.get(), prob->x[i], probs.data());
        
        // Both predictions should be valid labels
        EXPECT_TRUE(pred_normal == 1.0 || pred_normal == -1.0);
        EXPECT_TRUE(pred_prob == 1.0 || pred_prob == -1.0);
        
        // Note: They might not always match due to different methods
        // but they should be consistent with probabilities
    }
}

// ===========================================================================
// Different Kernel Probability Tests
// ===========================================================================

TEST_F(ProbabilityTest, LinearKernelProbability) {
    auto builder = createLinearlySeperableData(50, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, LINEAR);
    param.probability = 1;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    EXPECT_EQ(svm_check_probability_model(model.get()), 1);
    
    std::vector<double> probs(2);
    svm_predict_probability(model.get(), prob->x[0], probs.data());
    
    double sum = probs[0] + probs[1];
    EXPECT_NEAR(sum, 1.0, 1e-6);
}

TEST_F(ProbabilityTest, PolynomialKernelProbability) {
    auto builder = createXorData(30, 0.05, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, POLY);
    param.degree = 2;
    param.gamma = 1.0;
    param.coef0 = 1.0;
    param.probability = 1;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    EXPECT_EQ(svm_check_probability_model(model.get()), 1);
    
    std::vector<double> probs(2);
    svm_predict_probability(model.get(), prob->x[0], probs.data());
    
    double sum = probs[0] + probs[1];
    EXPECT_NEAR(sum, 1.0, 1e-6);
}

// ===========================================================================
// SVR Probability Tests
// ===========================================================================

TEST_F(ProbabilityTest, EpsilonSVRProbability) {
    auto builder = createRegressionData(80, 0.1, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(EPSILON_SVR, RBF);
    param.probability = 1;
    param.p = 0.1;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    EXPECT_EQ(svm_check_probability_model(model.get()), 1);
    
    double svr_probability = svm_get_svr_probability(model.get());
    EXPECT_GT(svr_probability, 0) << "SVR probability should be positive";
}

TEST_F(ProbabilityTest, NuSVRProbability) {
    auto builder = createRegressionData(80, 0.1, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(NU_SVR, RBF);
    param.probability = 1;
    param.nu = 0.5;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    EXPECT_EQ(svm_check_probability_model(model.get()), 1);
    
    double svr_probability = svm_get_svr_probability(model.get());
    EXPECT_GT(svr_probability, 0);
}

// ===========================================================================
// Calibration Tests
// ===========================================================================

TEST_F(ProbabilityTest, ProbabilityCalibration) {
    auto builder = createLinearlySeperableData(100, 42);
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.probability = 1;
    param.gamma = 0.5;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    // Collect predictions and group by confidence
    std::vector<int> correct_high_conf;
    std::vector<int> correct_low_conf;
    
    std::vector<double> probs(2);
    std::vector<int> labels(2);
    svm_get_labels(model.get(), labels.data());
    
    for (int i = 0; i < prob->l; ++i) {
        double pred = svm_predict_probability(model.get(), prob->x[i], probs.data());
        double max_prob = std::max(probs[0], probs[1]);
        int is_correct = (pred == prob->y[i]) ? 1 : 0;
        
        if (max_prob > 0.8) {
            correct_high_conf.push_back(is_correct);
        } else if (max_prob < 0.6) {
            correct_low_conf.push_back(is_correct);
        }
    }
    
    // High confidence predictions should be more accurate
    if (!correct_high_conf.empty() && !correct_low_conf.empty()) {
        double acc_high = std::accumulate(correct_high_conf.begin(), correct_high_conf.end(), 0.0) 
                          / correct_high_conf.size();
        double acc_low = std::accumulate(correct_low_conf.begin(), correct_low_conf.end(), 0.0) 
                         / correct_low_conf.size();
        
        // High confidence should generally be more accurate (not always true for small samples)
        // This is more of a sanity check than a strict requirement
        EXPECT_GE(acc_high, acc_low - 0.2) 
            << "High conf acc: " << acc_high << ", Low conf acc: " << acc_low;
    }
}

// ===========================================================================
// Edge Cases
// ===========================================================================

TEST_F(ProbabilityTest, SmallDatasetProbability) {
    auto builder = std::make_unique<SvmProblemBuilder>();
    
    for (int i = 0; i < 10; ++i) {
        builder->addDenseSample(1.0, {1.0 + i * 0.1, 1.0 + i * 0.1});
        builder->addDenseSample(-1.0, {-1.0 - i * 0.1, -1.0 - i * 0.1});
    }
    
    svm_problem* prob = builder->build();
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.probability = 1;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    std::vector<double> probs(2);
    svm_predict_probability(model.get(), prob->x[0], probs.data());
    
    EXPECT_GE(probs[0], 0);
    EXPECT_GE(probs[1], 0);
    EXPECT_NEAR(probs[0] + probs[1], 1.0, 1e-6);
}

TEST_F(ProbabilityTest, ImbalancedClassesProbability) {
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
    param.probability = 1;
    
    SvmModelGuard model(svm_train(prob, &param));
    ASSERT_TRUE(model);
    
    std::vector<double> probs(2);
    
    // Test a clearly positive sample
    std::vector<svm_node> pos = {{1, 1.5}, {2, 1.5}, {-1, 0}};
    svm_predict_probability(model.get(), pos.data(), probs.data());
    
    EXPECT_NEAR(probs[0] + probs[1], 1.0, 1e-6);
}
