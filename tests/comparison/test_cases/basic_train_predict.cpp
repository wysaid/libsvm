// Test case: Basic train and predict
// This file will be compiled twice: once for current version, once for upstream
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "svm.h"

// Helper function to create a simple 2D classification problem
struct svm_problem* create_simple_problem() {
    static struct svm_problem prob;
    static std::vector<double> y_data = {1, 1, -1, -1};
    static std::vector<std::vector<svm_node>> x_data = {
        {{0, 1.0}, {1, 2.0}, {-1, 0}},
        {{0, 2.0}, {1, 3.0}, {-1, 0}},
        {{0, -1.0}, {1, -2.0}, {-1, 0}},
        {{0, -2.0}, {1, -3.0}, {-1, 0}}
    };
    static std::vector<svm_node*> x_ptrs(4);
    
    for (size_t i = 0; i < x_data.size(); ++i) {
        x_ptrs[i] = x_data[i].data();
    }
    
    prob.l = 4;
    prob.y = y_data.data();
    prob.x = x_ptrs.data();
    
    return &prob;
}

// Helper to create default parameters
struct svm_parameter create_default_param() {
    struct svm_parameter param;
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.degree = 3;
    param.gamma = 0.5;
    param.coef0 = 0;
    param.cache_size = 100;
    param.eps = 1e-3;
    param.C = 1;
    param.nr_weight = 0;
    param.weight_label = nullptr;
    param.weight = nullptr;
    param.nu = 0.5;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    
    return param;
}

int main() {
    // Suppress libsvm output
    svm_set_print_string_function([](const char*){});
    
    // Create problem
    struct svm_problem* prob = create_simple_problem();
    
    // Create parameters
    struct svm_parameter param = create_default_param();
    
    // Train model
    struct svm_model* model = svm_train(prob, &param);
    
    if (!model) {
        std::cerr << "ERROR: Training failed" << std::endl;
        return 1;
    }
    
    // Output model info
    std::cout << "version:" << libsvm_version << std::endl;
    std::cout << "nr_class:" << svm_get_nr_class(model) << std::endl;
    std::cout << "nr_sv:" << svm_get_nr_sv(model) << std::endl;
    
    // Make predictions
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < prob->l; ++i) {
        double pred = svm_predict(model, prob->x[i]);
        std::cout << "pred_" << i << ":" << pred << std::endl;
    }
    
    // Cleanup
    svm_free_and_destroy_model(&model);
    
    return 0;
}
