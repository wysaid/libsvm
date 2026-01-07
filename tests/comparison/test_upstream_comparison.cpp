/**
 * @file test_upstream_comparison.cpp
 * @brief Comparison tests between this fork and upstream libsvm
 * 
 * This test compares the results of the current fork against the upstream
 * libsvm implementation to ensure compatibility and correctness.
 * 
 * To run these tests:
 * 1. Extract upstream code: git read-tree --prefix=tests/upstream_libsvm -u upstream
 * 2. Configure with: cmake -DLIBSVM_BUILD_TESTS=ON -DLIBSVM_BUILD_UPSTREAM_COMPARISON=ON ..
 * 3. Build and run tests
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <string>

// Include current fork's header
#include "svm.h"

// Include upstream header with namespace protection
namespace upstream {
    // Rename to avoid conflicts
    #define svm_node svm_node_up
    #define svm_problem svm_problem_up
    #define svm_parameter svm_parameter_up
    #define svm_model svm_model_up
    #define svm_train svm_train_up
    #define svm_predict svm_predict_up
    #define svm_predict_values svm_predict_values_up
    #define svm_cross_validation svm_cross_validation_up
    #define svm_save_model svm_save_model_up
    #define svm_load_model svm_load_model_up
    #define svm_free_and_destroy_model svm_free_and_destroy_model_up
    #define svm_free_model_content svm_free_model_content_up
    #define svm_destroy_param svm_destroy_param_up
    #define svm_check_parameter svm_check_parameter_up
    #define svm_check_probability_model svm_check_probability_model_up
    #define svm_get_svm_type svm_get_svm_type_up
    #define svm_get_nr_class svm_get_nr_class_up
    #define svm_get_nr_sv svm_get_nr_sv_up
    #define svm_get_labels svm_get_labels_up
    #define svm_get_sv_indices svm_get_sv_indices_up
    #define svm_get_svr_probability svm_get_svr_probability_up
    #define svm_set_print_string_function svm_set_print_string_function_up
    #define svm_predict_probability svm_predict_probability_up
    #define libsvm_version libsvm_version_up
    
    // Include upstream implementation
    // Note: This will be available when upstream_libsvm is extracted
    #ifdef UPSTREAM_AVAILABLE
    extern "C" {
        #include "svm.h"  // upstream_libsvm is in include path
    }
    #endif
}

#include "test_utils.h"

using namespace libsvm_test;

// Check if upstream is available
#ifdef UPSTREAM_AVAILABLE
#define UPSTREAM_TEST(test_name) TEST_F(UpstreamComparisonTest, test_name)
#else
#define UPSTREAM_TEST(test_name) TEST_F(UpstreamComparisonTest, DISABLED_##test_name)
#endif

class UpstreamComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        suppressOutput();
        #ifdef UPSTREAM_AVAILABLE
        upstream::svm_set_print_string_function_up([](const char*){});
        #endif
    }
    
    void TearDown() override {
        restoreOutput();
    }
    
    // Convert fork's svm_node to upstream format
    #ifdef UPSTREAM_AVAILABLE
    std::vector<upstream::svm_node_up> toUpstream(const svm_node* nodes) {
        std::vector<upstream::svm_node_up> result;
        int i = 0;
        while (nodes[i].index != -1) {
            upstream::svm_node_up node;
            node.index = nodes[i].index;
            node.value = nodes[i].value;
            result.push_back(node);
            ++i;
        }
        result.push_back({-1, 0});
        return result;
    }
    
    // Create parallel problems for both implementations
    struct ParallelProblem {
        std::unique_ptr<SvmProblemBuilder> fork_builder;
        svm_problem* fork_prob;
        
        upstream::svm_problem_up up_prob;
        std::vector<double> up_y;
        std::vector<std::vector<upstream::svm_node_up>> up_nodes;
        std::vector<upstream::svm_node_up*> up_x;
    };
    
    ParallelProblem createParallelLinearData(int n_samples, unsigned int seed) {
        ParallelProblem pp;
        
        pp.fork_builder = createLinearlySeperableData(n_samples, seed);
        pp.fork_prob = pp.fork_builder->build();
        
        // Copy to upstream format
        pp.up_y.resize(pp.fork_prob->l);
        pp.up_nodes.resize(pp.fork_prob->l);
        pp.up_x.resize(pp.fork_prob->l);
        
        for (int i = 0; i < pp.fork_prob->l; ++i) {
            pp.up_y[i] = pp.fork_prob->y[i];
            pp.up_nodes[i] = toUpstream(pp.fork_prob->x[i]);
            pp.up_x[i] = pp.up_nodes[i].data();
        }
        
        pp.up_prob.l = pp.fork_prob->l;
        pp.up_prob.y = pp.up_y.data();
        pp.up_prob.x = pp.up_x.data();
        
        return pp;
    }
    
    // Create parallel parameters
    upstream::svm_parameter_up toUpstreamParam(const svm_parameter& param) {
        upstream::svm_parameter_up up;
        up.svm_type = param.svm_type;
        up.kernel_type = param.kernel_type;
        up.degree = param.degree;
        up.gamma = param.gamma;
        up.coef0 = param.coef0;
        up.cache_size = param.cache_size;
        up.eps = param.eps;
        up.C = param.C;
        up.nr_weight = param.nr_weight;
        up.weight_label = param.weight_label;
        up.weight = param.weight;
        up.nu = param.nu;
        up.p = param.p;
        up.shrinking = param.shrinking;
        up.probability = param.probability;
        return up;
    }
    #endif
};

// ===========================================================================
// Basic Compatibility Tests
// ===========================================================================

TEST_F(UpstreamComparisonTest, VersionCheck) {
    // Just check that version is defined
    EXPECT_GT(libsvm_version, 0);
    EXPECT_GE(libsvm_version, 330);  // At least version 3.30
}

UPSTREAM_TEST(SamePredictionsBinaryClassification) {
    #ifdef UPSTREAM_AVAILABLE
    auto pp = createParallelLinearData(50, 42);
    
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.5;
    
    auto up_param = toUpstreamParam(param);
    
    // Train both models
    svm_model* fork_model = svm_train(pp.fork_prob, &param);
    upstream::svm_model_up* up_model = upstream::svm_train_up(&pp.up_prob, &up_param);
    
    ASSERT_NE(fork_model, nullptr);
    ASSERT_NE(up_model, nullptr);
    
    // Compare predictions
    int match_count = 0;
    for (int i = 0; i < pp.fork_prob->l; ++i) {
        double fork_pred = svm_predict(fork_model, pp.fork_prob->x[i]);
        double up_pred = upstream::svm_predict_up(up_model, pp.up_x[i]);
        
        if (fork_pred == up_pred) {
            ++match_count;
        }
    }
    
    double match_rate = static_cast<double>(match_count) / pp.fork_prob->l;
    EXPECT_GT(match_rate, 0.99) << "Prediction match rate: " << match_rate * 100 << "%";
    
    // Cleanup
    svm_free_and_destroy_model(&fork_model);
    upstream::svm_free_and_destroy_model_up(&up_model);
    #endif
}

UPSTREAM_TEST(SamePredictionsLinearKernel) {
    #ifdef UPSTREAM_AVAILABLE
    auto pp = createParallelLinearData(50, 42);
    
    svm_parameter param = getDefaultParameter(C_SVC, LINEAR);
    auto up_param = toUpstreamParam(param);
    
    svm_model* fork_model = svm_train(pp.fork_prob, &param);
    upstream::svm_model_up* up_model = upstream::svm_train_up(&pp.up_prob, &up_param);
    
    ASSERT_NE(fork_model, nullptr);
    ASSERT_NE(up_model, nullptr);
    
    // Linear kernel should give identical results
    for (int i = 0; i < pp.fork_prob->l; ++i) {
        double fork_pred = svm_predict(fork_model, pp.fork_prob->x[i]);
        double up_pred = upstream::svm_predict_up(up_model, pp.up_x[i]);
        
        EXPECT_DOUBLE_EQ(fork_pred, up_pred) << "Mismatch at sample " << i;
    }
    
    svm_free_and_destroy_model(&fork_model);
    upstream::svm_free_and_destroy_model_up(&up_model);
    #endif
}

UPSTREAM_TEST(SameNumberOfSupportVectors) {
    #ifdef UPSTREAM_AVAILABLE
    auto pp = createParallelLinearData(50, 42);
    
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.5;
    
    auto up_param = toUpstreamParam(param);
    
    svm_model* fork_model = svm_train(pp.fork_prob, &param);
    upstream::svm_model_up* up_model = upstream::svm_train_up(&pp.up_prob, &up_param);
    
    int fork_sv = svm_get_nr_sv(fork_model);
    int up_sv = upstream::svm_get_nr_sv_up(up_model);
    
    EXPECT_EQ(fork_sv, up_sv);
    
    svm_free_and_destroy_model(&fork_model);
    upstream::svm_free_and_destroy_model_up(&up_model);
    #endif
}

UPSTREAM_TEST(SameDecisionValues) {
    #ifdef UPSTREAM_AVAILABLE
    auto pp = createParallelLinearData(50, 42);
    
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.5;
    
    auto up_param = toUpstreamParam(param);
    
    svm_model* fork_model = svm_train(pp.fork_prob, &param);
    upstream::svm_model_up* up_model = upstream::svm_train_up(&pp.up_prob, &up_param);
    
    for (int i = 0; i < std::min(10, pp.fork_prob->l); ++i) {
        double fork_dv, up_dv;
        
        svm_predict_values(fork_model, pp.fork_prob->x[i], &fork_dv);
        upstream::svm_predict_values_up(up_model, pp.up_x[i], &up_dv);
        
        EXPECT_NEAR(fork_dv, up_dv, 1e-6) << "Decision value mismatch at sample " << i;
    }
    
    svm_free_and_destroy_model(&fork_model);
    upstream::svm_free_and_destroy_model_up(&up_model);
    #endif
}

UPSTREAM_TEST(SameCrossValidationResults) {
    #ifdef UPSTREAM_AVAILABLE
    auto pp = createParallelLinearData(100, 42);
    
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.5;
    
    auto up_param = toUpstreamParam(param);
    
    std::vector<double> fork_target(pp.fork_prob->l);
    std::vector<double> up_target(pp.fork_prob->l);
    
    svm_cross_validation(pp.fork_prob, &param, 5, fork_target.data());
    upstream::svm_cross_validation_up(&pp.up_prob, &up_param, 5, up_target.data());
    
    // Calculate accuracies
    int fork_correct = 0, up_correct = 0;
    for (int i = 0; i < pp.fork_prob->l; ++i) {
        if (fork_target[i] == pp.fork_prob->y[i]) ++fork_correct;
        if (up_target[i] == pp.up_prob.y[i]) ++up_correct;
    }
    
    double fork_acc = static_cast<double>(fork_correct) / pp.fork_prob->l;
    double up_acc = static_cast<double>(up_correct) / pp.fork_prob->l;
    
    // Accuracies should be very close
    EXPECT_NEAR(fork_acc, up_acc, 0.05) 
        << "Fork CV accuracy: " << fork_acc << ", Upstream: " << up_acc;
    #endif
}

UPSTREAM_TEST(SameModelSaveLoadFormat) {
    #ifdef UPSTREAM_AVAILABLE
    auto pp = createParallelLinearData(30, 42);
    
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.5;
    
    auto up_param = toUpstreamParam(param);
    
    svm_model* fork_model = svm_train(pp.fork_prob, &param);
    
    // Save fork model
    std::string fork_path = getTempFilePath("_fork.model");
    svm_save_model(fork_path.c_str(), fork_model);
    
    // Try to load with upstream (tests format compatibility)
    upstream::svm_model_up* loaded_by_upstream = upstream::svm_load_model_up(fork_path.c_str());
    
    if (loaded_by_upstream) {
        // If loading succeeded, predictions should match
        for (int i = 0; i < std::min(10, pp.fork_prob->l); ++i) {
            double fork_pred = svm_predict(fork_model, pp.fork_prob->x[i]);
            double up_loaded_pred = upstream::svm_predict_up(loaded_by_upstream, pp.up_x[i]);
            
            EXPECT_DOUBLE_EQ(fork_pred, up_loaded_pred);
        }
        
        upstream::svm_free_and_destroy_model_up(&loaded_by_upstream);
    }
    
    svm_free_and_destroy_model(&fork_model);
    deleteTempFile(fork_path);
    #endif
}

// ===========================================================================
// Regression Comparison Tests
// ===========================================================================

UPSTREAM_TEST(SameRegressionPredictions) {
    #ifdef UPSTREAM_AVAILABLE
    // Create regression data
    auto fork_builder = createRegressionData(50, 0.1, 42);
    auto fork_prob = fork_builder->build();
    
    // Copy to upstream format
    std::vector<double> up_y(fork_prob->l);
    std::vector<std::vector<upstream::svm_node_up>> up_nodes(fork_prob->l);
    std::vector<upstream::svm_node_up*> up_x(fork_prob->l);
    
    for (int i = 0; i < fork_prob->l; ++i) {
        up_y[i] = fork_prob->y[i];
        up_nodes[i] = toUpstream(fork_prob->x[i]);
        up_x[i] = up_nodes[i].data();
    }
    
    upstream::svm_problem_up up_prob;
    up_prob.l = fork_prob->l;
    up_prob.y = up_y.data();
    up_prob.x = up_x.data();
    
    svm_parameter param = getDefaultParameter(EPSILON_SVR, RBF);
    param.gamma = 0.5;
    param.p = 0.1;
    
    auto up_param = toUpstreamParam(param);
    
    svm_model* fork_model = svm_train(fork_prob, &param);
    upstream::svm_model_up* up_model = upstream::svm_train_up(&up_prob, &up_param);
    
    // Compare MSE
    double fork_mse = 0, up_mse = 0;
    for (int i = 0; i < fork_prob->l; ++i) {
        double fork_pred = svm_predict(fork_model, fork_prob->x[i]);
        double up_pred = upstream::svm_predict_up(up_model, up_x[i]);
        
        double fork_err = fork_pred - fork_prob->y[i];
        double up_err = up_pred - fork_prob->y[i];
        
        fork_mse += fork_err * fork_err;
        up_mse += up_err * up_err;
    }
    
    fork_mse /= fork_prob->l;
    up_mse /= fork_prob->l;
    
    EXPECT_NEAR(fork_mse, up_mse, 0.1) << "MSE difference too large";
    
    svm_free_and_destroy_model(&fork_model);
    upstream::svm_free_and_destroy_model_up(&up_model);
    #endif
}

// ===========================================================================
// Multi-class Comparison Tests
// ===========================================================================

UPSTREAM_TEST(SameMulticlassPredictions) {
    #ifdef UPSTREAM_AVAILABLE
    auto fork_builder = createMultiClassData(4, 30, 4, 42);
    auto fork_prob = fork_builder->build();
    
    // Copy to upstream format
    std::vector<double> up_y(fork_prob->l);
    std::vector<std::vector<upstream::svm_node_up>> up_nodes(fork_prob->l);
    std::vector<upstream::svm_node_up*> up_x(fork_prob->l);
    
    for (int i = 0; i < fork_prob->l; ++i) {
        up_y[i] = fork_prob->y[i];
        up_nodes[i] = toUpstream(fork_prob->x[i]);
        up_x[i] = up_nodes[i].data();
    }
    
    upstream::svm_problem_up up_prob;
    up_prob.l = fork_prob->l;
    up_prob.y = up_y.data();
    up_prob.x = up_x.data();
    
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.5;
    
    auto up_param = toUpstreamParam(param);
    
    svm_model* fork_model = svm_train(fork_prob, &param);
    upstream::svm_model_up* up_model = upstream::svm_train_up(&up_prob, &up_param);
    
    EXPECT_EQ(svm_get_nr_class(fork_model), 4);
    EXPECT_EQ(upstream::svm_get_nr_class_up(up_model), 4);
    
    int match_count = 0;
    for (int i = 0; i < fork_prob->l; ++i) {
        double fork_pred = svm_predict(fork_model, fork_prob->x[i]);
        double up_pred = upstream::svm_predict_up(up_model, up_x[i]);
        
        if (fork_pred == up_pred) ++match_count;
    }
    
    double match_rate = static_cast<double>(match_count) / fork_prob->l;
    EXPECT_GT(match_rate, 0.95);
    
    svm_free_and_destroy_model(&fork_model);
    upstream::svm_free_and_destroy_model_up(&up_model);
    #endif
}

// ===========================================================================
// Performance Comparison Tests (Timing)
// ===========================================================================

UPSTREAM_TEST(SimilarTrainingTime) {
    #ifdef UPSTREAM_AVAILABLE
    auto pp = createParallelLinearData(200, 42);
    
    svm_parameter param = getDefaultParameter(C_SVC, RBF);
    param.gamma = 0.5;
    
    auto up_param = toUpstreamParam(param);
    
    // Time fork training
    auto fork_start = std::chrono::high_resolution_clock::now();
    svm_model* fork_model = svm_train(pp.fork_prob, &param);
    auto fork_end = std::chrono::high_resolution_clock::now();
    
    // Time upstream training
    auto up_start = std::chrono::high_resolution_clock::now();
    upstream::svm_model_up* up_model = upstream::svm_train_up(&pp.up_prob, &up_param);
    auto up_end = std::chrono::high_resolution_clock::now();
    
    auto fork_duration = std::chrono::duration_cast<std::chrono::milliseconds>(fork_end - fork_start);
    auto up_duration = std::chrono::duration_cast<std::chrono::milliseconds>(up_end - up_start);
    
    // Fork should not be significantly slower (within 50% overhead)
    double ratio = static_cast<double>(fork_duration.count()) / 
                   std::max(1LL, static_cast<long long>(up_duration.count()));
    EXPECT_LT(ratio, 1.5) << "Fork: " << fork_duration.count() << "ms, Upstream: " << up_duration.count() << "ms";
    
    svm_free_and_destroy_model(&fork_model);
    upstream::svm_free_and_destroy_model_up(&up_model);
    #endif
}

// If upstream is not available, provide a skip message
TEST_F(UpstreamComparisonTest, UpstreamAvailabilityCheck) {
    #ifndef UPSTREAM_AVAILABLE
    GTEST_SKIP() << "Upstream libsvm not available. "
                 << "To enable these tests, run: "
                 << "git read-tree --prefix=tests/upstream_libsvm -u upstream";
    #else
    SUCCEED() << "Upstream libsvm is available for comparison tests";
    #endif
}
