// Test case: Model save and load
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdio>
#include <iomanip>
#include "svm.h"

// Helper to create simple problem
struct svm_problem* create_problem() {
    static struct svm_problem prob;
    static std::vector<double> y_data = {1, 1, -1, -1, 1, -1};
    static std::vector<std::vector<svm_node>> x_data = {
        {{0, 1.0}, {1, 1.0}, {-1, 0}},
        {{0, 2.0}, {1, 2.0}, {-1, 0}},
        {{0, -1.0}, {1, -1.0}, {-1, 0}},
        {{0, -2.0}, {1, -2.0}, {-1, 0}},
        {{0, 1.5}, {1, 1.5}, {-1, 0}},
        {{0, -1.5}, {1, -1.5}, {-1, 0}}
    };
    static std::vector<svm_node*> x_ptrs(6);
    
    for (size_t i = 0; i < x_data.size(); ++i) {
        x_ptrs[i] = x_data[i].data();
    }
    
    prob.l = 6;
    prob.y = y_data.data();
    prob.x = x_ptrs.data();
    
    return &prob;
}

struct svm_parameter create_param() {
    struct svm_parameter param;
    param.svm_type = C_SVC;
    param.kernel_type = LINEAR;
    param.degree = 3;
    param.gamma = 0;
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
    svm_set_print_string_function([](const char*){});
    
    // Create and train model
    struct svm_problem* prob = create_problem();
    struct svm_parameter param = create_param();
    struct svm_model* model = svm_train(prob, &param);
    
    if (!model) {
        std::cerr << "ERROR: Training failed" << std::endl;
        return 1;
    }
    
    // Save model
    const char* model_file = "/tmp/libsvm_test_model.txt";
    if (svm_save_model(model_file, model) != 0) {
        std::cerr << "ERROR: Failed to save model" << std::endl;
        svm_free_and_destroy_model(&model);
        return 1;
    }
    
    // Make predictions with original model
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "version:" << libsvm_version << std::endl;
    
    std::vector<double> orig_predictions;
    for (int i = 0; i < prob->l; ++i) {
        double pred = svm_predict(model, prob->x[i]);
        orig_predictions.push_back(pred);
    }
    
    // Free original model
    svm_free_and_destroy_model(&model);
    
    // Load model
    struct svm_model* loaded_model = svm_load_model(model_file);
    if (!loaded_model) {
        std::cerr << "ERROR: Failed to load model" << std::endl;
        std::remove(model_file);
        return 1;
    }
    
    // Make predictions with loaded model
    for (int i = 0; i < prob->l; ++i) {
        double pred = svm_predict(loaded_model, prob->x[i]);
        std::cout << "pred_" << i << ":" << pred << std::endl;
        
        // Verify consistency
        if (pred != orig_predictions[i]) {
            std::cerr << "ERROR: Prediction mismatch at " << i << std::endl;
            svm_free_and_destroy_model(&loaded_model);
            std::remove(model_file);
            return 1;
        }
    }
    
    // Output model info
    std::cout << "nr_class:" << svm_get_nr_class(loaded_model) << std::endl;
    std::cout << "nr_sv:" << svm_get_nr_sv(loaded_model) << std::endl;
    
    // Cleanup
    svm_free_and_destroy_model(&loaded_model);
    std::remove(model_file);
    
    return 0;
}
