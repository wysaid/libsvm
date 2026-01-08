/**
 * @file test_opencv_comparison.cpp
 * @brief Comparison tests between LibSVM and OpenCV's SVM implementation
 * 
 * OpenCV's ml module includes an SVM implementation that is compatible with
 * LibSVM. This test compares both implementations on the same data.
 * 
 * OpenCV's SVM is based on LibSVM but with some differences:
 * - Uses cv::Mat for data representation
 * - Integrated into OpenCV's machine learning pipeline
 * - Same kernel types and parameters available
 * 
 * To run these tests:
 * 1. Install OpenCV with ml module
 * 2. Configure with: cmake -DLIBSVM_BUILD_TESTS=ON -DLIBSVM_BUILD_OPENCV_COMPARISON=ON ..
 * 3. Build and run tests
 */

#include <gtest/gtest.h>
#include "svm.h"
#include "test_utils.h"
#include <vector>
#include <cmath>
#include <chrono>

#ifdef OPENCV_AVAILABLE
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#endif

using namespace libsvm_test;

class OpenCVComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        suppressOutput();
    }
    
    void TearDown() override {
        restoreOutput();
    }
    
#ifdef OPENCV_AVAILABLE
    // Convert LibSVM problem to OpenCV format
    struct OpenCVData {
        cv::Mat samples;
        cv::Mat labels;
    };
    
    OpenCVData toOpenCV(svm_problem* prob, int max_feature_index = -1) {
        OpenCVData data;
        
        // Find max feature index if not specified
        if (max_feature_index < 0) {
            max_feature_index = 0;
            for (int i = 0; i < prob->l; ++i) {
                int j = 0;
                while (prob->x[i][j].index != -1) {
                    max_feature_index = std::max(max_feature_index, prob->x[i][j].index);
                    ++j;
                }
            }
        }
        
        data.samples = cv::Mat::zeros(prob->l, max_feature_index, CV_32F);
        data.labels = cv::Mat(prob->l, 1, CV_32S);
        
        for (int i = 0; i < prob->l; ++i) {
            data.labels.at<int>(i) = static_cast<int>(prob->y[i]);
            
            int j = 0;
            while (prob->x[i][j].index != -1) {
                int idx = prob->x[i][j].index - 1;  // 0-based indexing
                if (idx < max_feature_index) {
                    data.samples.at<float>(i, idx) = static_cast<float>(prob->x[i][j].value);
                }
                ++j;
            }
        }
        
        return data;
    }
    
    // Create OpenCV SVM with LibSVM-compatible parameters
    cv::Ptr<cv::ml::SVM> createOpenCVSVM(const svm_parameter& param) {
        auto svm = cv::ml::SVM::create();
        
        // Set SVM type
        switch (param.svm_type) {
            case C_SVC:
                svm->setType(cv::ml::SVM::C_SVC);
                break;
            case NU_SVC:
                svm->setType(cv::ml::SVM::NU_SVC);
                break;
            case ONE_CLASS:
                svm->setType(cv::ml::SVM::ONE_CLASS);
                break;
            case EPSILON_SVR:
                svm->setType(cv::ml::SVM::EPS_SVR);
                break;
            case NU_SVR:
                svm->setType(cv::ml::SVM::NU_SVR);
                break;
        }
        
        // Set kernel type
        switch (param.kernel_type) {
            case LINEAR:
                svm->setKernel(cv::ml::SVM::LINEAR);
                break;
            case POLY:
                svm->setKernel(cv::ml::SVM::POLY);
                svm->setDegree(param.degree);
                svm->setCoef0(param.coef0);
                break;
            case RBF:
                svm->setKernel(cv::ml::SVM::RBF);
                break;
            case SIGMOID:
                svm->setKernel(cv::ml::SVM::SIGMOID);
                svm->setCoef0(param.coef0);
                break;
        }
        
        // Set common parameters
        svm->setGamma(param.gamma);
        svm->setC(param.C);
        svm->setNu(param.nu);
        svm->setP(param.p);
        
        // Termination criteria
        svm->setTermCriteria(cv::TermCriteria(
            cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 
            10000, 
            param.eps
        ));
        
        return svm;
    }
#endif
};

#ifdef OPENCV_AVAILABLE
#define OPENCV_TEST(test_name) TEST_F(OpenCVComparisonTest, test_name)
#else
#define OPENCV_TEST(test_name) TEST_F(OpenCVComparisonTest, DISABLED_##test_name)
#endif

// ===========================================================================
// Basic Comparison Tests
// ===========================================================================

OPENCV_TEST(BinaryClassificationComparison) {
#ifdef OPENCV_AVAILABLE
    auto builder = createLinearlySeperableData(100, 42);
    svm_problem* prob = builder->build();
    
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.5;
    param.C = 1.0;
    
    // Train LibSVM
    SvmModelGuard libsvm_model(svm_train(prob, &param));
    ASSERT_TRUE(libsvm_model);
    
    // Train OpenCV SVM
    auto cv_data = toOpenCV(prob);
    auto cv_svm = createOpenCVSVM(param);
    bool trained = cv_svm->train(cv_data.samples, cv::ml::ROW_SAMPLE, cv_data.labels);
    ASSERT_TRUE(trained);
    
    // Compare accuracy on training data
    int libsvm_correct = 0, opencv_correct = 0;
    
    for (int i = 0; i < prob->l; ++i) {
        // LibSVM prediction
        double libsvm_pred = svm_predict(libsvm_model.get(), prob->x[i]);
        
        // OpenCV prediction
        cv::Mat sample = cv_data.samples.row(i);
        float opencv_pred = cv_svm->predict(sample);
        
        if (libsvm_pred == prob->y[i]) ++libsvm_correct;
        if (opencv_pred == static_cast<float>(prob->y[i])) ++opencv_correct;
    }
    
    double libsvm_acc = static_cast<double>(libsvm_correct) / prob->l;
    double opencv_acc = static_cast<double>(opencv_correct) / prob->l;
    
    // Both should have good accuracy on linearly separable data
    EXPECT_GT(libsvm_acc, 0.9) << "LibSVM accuracy: " << libsvm_acc * 100 << "%";
    EXPECT_GT(opencv_acc, 0.9) << "OpenCV accuracy: " << opencv_acc * 100 << "%";
    
    // Accuracies should be similar
    EXPECT_NEAR(libsvm_acc, opencv_acc, 0.1) 
        << "LibSVM: " << libsvm_acc * 100 << "%, OpenCV: " << opencv_acc * 100 << "%";
#endif
}

OPENCV_TEST(LinearKernelComparison) {
#ifdef OPENCV_AVAILABLE
    auto builder = createLinearlySeperableData(100, 42);
    svm_problem* prob = builder->build();
    
    svm_parameter param = getDefaultParameter(C_SVC, LINEAR);
    param.C = 1.0;
    
    SvmModelGuard libsvm_model(svm_train(prob, &param));
    ASSERT_TRUE(libsvm_model);
    
    auto cv_data = toOpenCV(prob);
    auto cv_svm = createOpenCVSVM(param);
    cv_svm->train(cv_data.samples, cv::ml::ROW_SAMPLE, cv_data.labels);
    
    // Both should achieve near-perfect accuracy on linearly separable data
    int libsvm_correct = 0, opencv_correct = 0;
    
    for (int i = 0; i < prob->l; ++i) {
        double libsvm_pred = svm_predict(libsvm_model.get(), prob->x[i]);
        float opencv_pred = cv_svm->predict(cv_data.samples.row(i));
        
        if (libsvm_pred == prob->y[i]) ++libsvm_correct;
        if (opencv_pred == static_cast<float>(prob->y[i])) ++opencv_correct;
    }
    
    double libsvm_acc = static_cast<double>(libsvm_correct) / prob->l;
    double opencv_acc = static_cast<double>(opencv_correct) / prob->l;
    
    EXPECT_GT(libsvm_acc, 0.95);
    EXPECT_GT(opencv_acc, 0.95);
#endif
}

OPENCV_TEST(RBFKernelXORComparison) {
#ifdef OPENCV_AVAILABLE
    auto builder = createXorData(30, 0.05, 42);
    svm_problem* prob = builder->build();
    
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 1.0;
    param.C = 10.0;
    
    SvmModelGuard libsvm_model(svm_train(prob, &param));
    ASSERT_TRUE(libsvm_model);
    
    auto cv_data = toOpenCV(prob);
    auto cv_svm = createOpenCVSVM(param);
    cv_svm->train(cv_data.samples, cv::ml::ROW_SAMPLE, cv_data.labels);
    
    int libsvm_correct = 0, opencv_correct = 0;
    
    for (int i = 0; i < prob->l; ++i) {
        double libsvm_pred = svm_predict(libsvm_model.get(), prob->x[i]);
        float opencv_pred = cv_svm->predict(cv_data.samples.row(i));
        
        if (libsvm_pred == prob->y[i]) ++libsvm_correct;
        if (opencv_pred == static_cast<float>(prob->y[i])) ++opencv_correct;
    }
    
    double libsvm_acc = static_cast<double>(libsvm_correct) / prob->l;
    double opencv_acc = static_cast<double>(opencv_correct) / prob->l;
    
    // Both should handle XOR pattern with RBF kernel
    EXPECT_GT(libsvm_acc, 0.8);
    EXPECT_GT(opencv_acc, 0.8);
#endif
}

// ===========================================================================
// Multi-class Comparison Tests
// ===========================================================================

OPENCV_TEST(MultiClassComparison) {
#ifdef OPENCV_AVAILABLE
    auto builder = createMultiClassData(4, 50, 4, 42);
    svm_problem* prob = builder->build();
    
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.5;
    param.C = 1.0;
    
    SvmModelGuard libsvm_model(svm_train(prob, &param));
    ASSERT_TRUE(libsvm_model);
    
    auto cv_data = toOpenCV(prob);
    auto cv_svm = createOpenCVSVM(param);
    cv_svm->train(cv_data.samples, cv::ml::ROW_SAMPLE, cv_data.labels);
    
    int libsvm_correct = 0, opencv_correct = 0;
    
    for (int i = 0; i < prob->l; ++i) {
        double libsvm_pred = svm_predict(libsvm_model.get(), prob->x[i]);
        float opencv_pred = cv_svm->predict(cv_data.samples.row(i));
        
        if (libsvm_pred == prob->y[i]) ++libsvm_correct;
        if (static_cast<int>(opencv_pred) == static_cast<int>(prob->y[i])) ++opencv_correct;
    }
    
    double libsvm_acc = static_cast<double>(libsvm_correct) / prob->l;
    double opencv_acc = static_cast<double>(opencv_correct) / prob->l;
    
    EXPECT_GT(libsvm_acc, 0.7);
    EXPECT_GT(opencv_acc, 0.7);
#endif
}

// ===========================================================================
// Regression Comparison Tests
// ===========================================================================

OPENCV_TEST(RegressionComparison) {
#ifdef OPENCV_AVAILABLE
    auto builder = createRegressionData(100, 0.1, 42);
    svm_problem* prob = builder->build();
    
    svm_parameter param = getDefaultParameter(EPSILON_SVR, RBF);
    param.gamma = 0.5;
    param.p = 0.1;
    param.C = 10.0;
    
    SvmModelGuard libsvm_model(svm_train(prob, &param));
    ASSERT_TRUE(libsvm_model);
    
    // OpenCV regression needs float labels
    auto cv_data = toOpenCV(prob);
    cv::Mat float_labels(prob->l, 1, CV_32F);
    for (int i = 0; i < prob->l; ++i) {
        float_labels.at<float>(i) = static_cast<float>(prob->y[i]);
    }
    
    auto cv_svm = createOpenCVSVM(param);
    cv_svm->train(cv_data.samples, cv::ml::ROW_SAMPLE, float_labels);
    
    // Compare MSE
    double libsvm_mse = 0, opencv_mse = 0;
    
    for (int i = 0; i < prob->l; ++i) {
        double libsvm_pred = svm_predict(libsvm_model.get(), prob->x[i]);
        float opencv_pred = cv_svm->predict(cv_data.samples.row(i));
        
        double libsvm_err = libsvm_pred - prob->y[i];
        double opencv_err = opencv_pred - prob->y[i];
        
        libsvm_mse += libsvm_err * libsvm_err;
        opencv_mse += opencv_err * opencv_err;
    }
    
    libsvm_mse /= prob->l;
    opencv_mse /= prob->l;
    
    // Both should achieve reasonable MSE
    EXPECT_LT(libsvm_mse, 2.0) << "LibSVM MSE: " << libsvm_mse;
    EXPECT_LT(opencv_mse, 2.0) << "OpenCV MSE: " << opencv_mse;
#endif
}

// ===========================================================================
// Performance Comparison Tests
// ===========================================================================

OPENCV_TEST(TrainingTimeComparison) {
#ifdef OPENCV_AVAILABLE
    auto builder = createLinearlySeperableData(500, 42);
    svm_problem* prob = builder->build();
    
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.5;
    
    auto cv_data = toOpenCV(prob);
    auto cv_svm = createOpenCVSVM(param);
    
    // Time LibSVM
    auto libsvm_start = std::chrono::high_resolution_clock::now();
    SvmModelGuard libsvm_model(svm_train(prob, &param));
    auto libsvm_end = std::chrono::high_resolution_clock::now();
    
    // Time OpenCV
    auto opencv_start = std::chrono::high_resolution_clock::now();
    cv_svm->train(cv_data.samples, cv::ml::ROW_SAMPLE, cv_data.labels);
    auto opencv_end = std::chrono::high_resolution_clock::now();
    
    auto libsvm_ms = std::chrono::duration_cast<std::chrono::milliseconds>(libsvm_end - libsvm_start).count();
    auto opencv_ms = std::chrono::duration_cast<std::chrono::milliseconds>(opencv_end - opencv_start).count();
    
    // Just log the times, don't fail on performance differences
    std::cout << "LibSVM training time: " << libsvm_ms << " ms" << std::endl;
    std::cout << "OpenCV training time: " << opencv_ms << " ms" << std::endl;
    
    SUCCEED();
#endif
}

OPENCV_TEST(PredictionTimeComparison) {
#ifdef OPENCV_AVAILABLE
    auto builder = createLinearlySeperableData(100, 42);
    svm_problem* prob = builder->build();
    
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.5;
    
    SvmModelGuard libsvm_model(svm_train(prob, &param));
    
    auto cv_data = toOpenCV(prob);
    auto cv_svm = createOpenCVSVM(param);
    cv_svm->train(cv_data.samples, cv::ml::ROW_SAMPLE, cv_data.labels);
    
    const int n_iterations = 1000;
    
    // Time LibSVM predictions
    auto libsvm_start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < n_iterations; ++iter) {
        for (int i = 0; i < prob->l; ++i) {
            svm_predict(libsvm_model.get(), prob->x[i]);
        }
    }
    auto libsvm_end = std::chrono::high_resolution_clock::now();
    
    // Time OpenCV predictions
    auto opencv_start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < n_iterations; ++iter) {
        for (int i = 0; i < prob->l; ++i) {
            cv_svm->predict(cv_data.samples.row(i));
        }
    }
    auto opencv_end = std::chrono::high_resolution_clock::now();
    
    auto libsvm_us = std::chrono::duration_cast<std::chrono::microseconds>(libsvm_end - libsvm_start).count();
    auto opencv_us = std::chrono::duration_cast<std::chrono::microseconds>(opencv_end - opencv_start).count();
    
    std::cout << "LibSVM prediction time (" << n_iterations * prob->l << " predictions): " 
              << libsvm_us / 1000.0 << " ms" << std::endl;
    std::cout << "OpenCV prediction time (" << n_iterations * prob->l << " predictions): " 
              << opencv_us / 1000.0 << " ms" << std::endl;
    
    SUCCEED();
#endif
}

// ===========================================================================
// Feature Comparison Tests
// ===========================================================================

OPENCV_TEST(SupportVectorCountComparison) {
#ifdef OPENCV_AVAILABLE
    auto builder = createLinearlySeperableData(100, 42);
    svm_problem* prob = builder->build();
    
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.5;
    param.C = 1.0;
    
    SvmModelGuard libsvm_model(svm_train(prob, &param));
    
    auto cv_data = toOpenCV(prob);
    auto cv_svm = createOpenCVSVM(param);
    cv_svm->train(cv_data.samples, cv::ml::ROW_SAMPLE, cv_data.labels);
    
    int libsvm_sv = svm_get_nr_sv(libsvm_model.get());
    int opencv_sv = cv_svm->getSupportVectors().rows;
    
    // Support vector counts should be similar
    // Allow some difference due to numerical precision
    EXPECT_NEAR(libsvm_sv, opencv_sv, std::max(5, libsvm_sv / 10))
        << "LibSVM SVs: " << libsvm_sv << ", OpenCV SVs: " << opencv_sv;
#endif
}

// ===========================================================================
// Edge Cases
// ===========================================================================

OPENCV_TEST(SmallDatasetComparison) {
#ifdef OPENCV_AVAILABLE
    auto builder = std::make_unique<SvmProblemBuilder>();
    
    for (int i = 0; i < 10; ++i) {
        builder->addDenseSample(1.0, {1.0 + i * 0.1, 1.0 + i * 0.1});
        builder->addDenseSample(-1.0, {-1.0 - i * 0.1, -1.0 - i * 0.1});
    }
    
    svm_problem* prob = builder->build();
    
    svm_parameter param = getDefaultParameter(C_SVC, LINEAR);
    
    SvmModelGuard libsvm_model(svm_train(prob, &param));
    
    auto cv_data = toOpenCV(prob);
    auto cv_svm = createOpenCVSVM(param);
    cv_svm->train(cv_data.samples, cv::ml::ROW_SAMPLE, cv_data.labels);
    
    // Both should handle small dataset
    ASSERT_TRUE(libsvm_model);
    EXPECT_GT(svm_get_nr_sv(libsvm_model.get()), 0);
#endif
}

OPENCV_TEST(HighDimensionalDataComparison) {
#ifdef OPENCV_AVAILABLE
    auto builder = std::make_unique<SvmProblemBuilder>();
    
    // 50-dimensional data
    for (int i = 0; i < 50; ++i) {
        std::vector<double> pos_features(50), neg_features(50);
        for (int j = 0; j < 50; ++j) {
            pos_features[j] = 0.5 + (i * j % 10) * 0.01;
            neg_features[j] = -0.5 - (i * j % 10) * 0.01;
        }
        builder->addDenseSample(1.0, pos_features);
        builder->addDenseSample(-1.0, neg_features);
    }
    
    svm_problem* prob = builder->build();
    
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.02;  // 1 / n_features
    
    SvmModelGuard libsvm_model(svm_train(prob, &param));
    
    auto cv_data = toOpenCV(prob, 50);
    auto cv_svm = createOpenCVSVM(param);
    cv_svm->train(cv_data.samples, cv::ml::ROW_SAMPLE, cv_data.labels);
    
    int libsvm_correct = 0, opencv_correct = 0;
    
    for (int i = 0; i < prob->l; ++i) {
        double libsvm_pred = svm_predict(libsvm_model.get(), prob->x[i]);
        float opencv_pred = cv_svm->predict(cv_data.samples.row(i));
        
        if (libsvm_pred == prob->y[i]) ++libsvm_correct;
        if (opencv_pred == static_cast<float>(prob->y[i])) ++opencv_correct;
    }
    
    double libsvm_acc = static_cast<double>(libsvm_correct) / prob->l;
    double opencv_acc = static_cast<double>(opencv_correct) / prob->l;
    
    EXPECT_GT(libsvm_acc, 0.8);
    EXPECT_GT(opencv_acc, 0.8);
#endif
}

// ===========================================================================
// Availability Check
// ===========================================================================

TEST_F(OpenCVComparisonTest, OpenCVAvailabilityCheck) {
#ifndef OPENCV_AVAILABLE
    GTEST_SKIP() << "OpenCV not available. "
                 << "To enable these tests, install OpenCV with ml module "
                 << "and configure with -DLIBSVM_BUILD_OPENCV_COMPARISON=ON";
#else
    // Check OpenCV version
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    SUCCEED() << "OpenCV is available for comparison tests";
#endif
}
