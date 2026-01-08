// Test case: Cross validation
#include <iostream>
#include <vector>
#include <iomanip>
#include "svm.h"

struct svm_problem* create_linear_problem() {
    static struct svm_problem prob;
    static std::vector<double> y_data = {1, 1, 1, -1, -1, -1, 1, 1, -1, -1};
    static std::vector<std::vector<svm_node>> x_data = {
        {{0, 1.0}, {1, 1.0}, {-1, 0}},
        {{0, 1.5}, {1, 1.5}, {-1, 0}},
        {{0, 2.0}, {1, 2.0}, {-1, 0}},
        {{0, -1.0}, {1, -1.0}, {-1, 0}},
        {{0, -1.5}, {1, -1.5}, {-1, 0}},
        {{0, -2.0}, {1, -2.0}, {-1, 0}},
        {{0, 1.2}, {1, 1.8}, {-1, 0}},
        {{0, 1.8}, {1, 1.2}, {-1, 0}},
        {{0, -1.2}, {1, -1.8}, {-1, 0}},
        {{0, -1.8}, {1, -1.2}, {-1, 0}}
    };
    static std::vector<svm_node*> x_ptrs(10);
    
    for (size_t i = 0; i < x_data.size(); ++i) {
        x_ptrs[i] = x_data[i].data();
    }
    
    prob.l = 10;
    prob.y = y_data.data();
    prob.x = x_ptrs.data();
    
    return &prob;
}

struct svm_parameter create_param() {
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
    svm_set_print_string_function([](const char*){});
    
    struct svm_problem* prob = create_linear_problem();
    struct svm_parameter param = create_param();
    
    std::vector<double> target(prob->l);
    svm_cross_validation(prob, &param, 5, target.data());
    
    std::cout << "version:" << libsvm_version << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    
    int correct = 0;
    for (int i = 0; i < prob->l; ++i) {
        std::cout << "cv_" << i << ":" << target[i] << std::endl;
        if (target[i] == prob->y[i]) {
            ++correct;
        }
    }
    
    double accuracy = static_cast<double>(correct) / prob->l;
    std::cout << "accuracy:" << accuracy << std::endl;
    
    return 0;
}
