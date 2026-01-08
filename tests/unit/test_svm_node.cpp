/**
 * @file test_svm_node.cpp
 * @brief Unit tests for svm_node structure
 */

#include <gtest/gtest.h>
#include "svm.h"
#include "test_utils.h"
#include <vector>
#include <climits>
#include <cfloat>

using namespace libsvm_test;

class SvmNodeTest : public ::testing::Test {
protected:
    void SetUp() override {
        suppressOutput();
    }
    
    void TearDown() override {
        restoreOutput();
    }
};

// Test svm_node basic structure
TEST_F(SvmNodeTest, BasicStructure) {
    svm_node node;
    node.index = 1;
    node.value = 0.5;
    
    EXPECT_EQ(node.index, 1);
    EXPECT_DOUBLE_EQ(node.value, 0.5);
}

// Test svm_node terminator convention
TEST_F(SvmNodeTest, TerminatorConvention) {
    svm_node terminator;
    terminator.index = -1;
    terminator.value = 0;
    
    EXPECT_EQ(terminator.index, -1);
    EXPECT_DOUBLE_EQ(terminator.value, 0.0);
}

// Test svm_node array (sparse vector representation)
TEST_F(SvmNodeTest, SparseVectorRepresentation) {
    // Sparse vector: {1: 0.5, 3: 1.0, 5: -0.5}
    std::vector<svm_node> nodes = {
        {1, 0.5},
        {3, 1.0},
        {5, -0.5},
        {-1, 0.0}  // terminator
    };
    
    EXPECT_EQ(nodes.size(), 4);
    EXPECT_EQ(nodes[0].index, 1);
    EXPECT_DOUBLE_EQ(nodes[0].value, 0.5);
    EXPECT_EQ(nodes[1].index, 3);
    EXPECT_DOUBLE_EQ(nodes[1].value, 1.0);
    EXPECT_EQ(nodes[2].index, 5);
    EXPECT_DOUBLE_EQ(nodes[2].value, -0.5);
    EXPECT_EQ(nodes[3].index, -1);
}

// Test svm_node with zero values (should be included in sparse format)
TEST_F(SvmNodeTest, SparseWithZeros) {
    // Explicitly including zeros
    std::vector<svm_node> nodes = {
        {1, 0.0},
        {2, 1.0},
        {-1, 0.0}
    };
    
    EXPECT_EQ(nodes.size(), 3);
    EXPECT_EQ(nodes[0].index, 1);
    EXPECT_DOUBLE_EQ(nodes[0].value, 0.0);
}

// Test svm_node with extreme values
TEST_F(SvmNodeTest, ExtremeValues) {
    svm_node node_max;
    node_max.index = INT_MAX;
    node_max.value = DBL_MAX;
    
    EXPECT_EQ(node_max.index, INT_MAX);
    EXPECT_DOUBLE_EQ(node_max.value, DBL_MAX);
    
    svm_node node_min;
    node_min.index = 1;
    node_min.value = DBL_MIN;
    
    EXPECT_EQ(node_min.index, 1);
    EXPECT_DOUBLE_EQ(node_min.value, DBL_MIN);
}

// Test svm_node with negative index (should only be -1 for terminator)
TEST_F(SvmNodeTest, NegativeIndex) {
    svm_node terminator;
    terminator.index = -1;
    
    // Only -1 should be used for terminator
    EXPECT_EQ(terminator.index, -1);
}

// Test creating an empty feature vector (just terminator)
TEST_F(SvmNodeTest, EmptyFeatureVector) {
    std::vector<svm_node> empty_vec = {{-1, 0.0}};
    
    EXPECT_EQ(empty_vec.size(), 1);
    EXPECT_EQ(empty_vec[0].index, -1);
}

// Test high-dimensional sparse vector
TEST_F(SvmNodeTest, HighDimensionalSparse) {
    std::vector<svm_node> sparse_vec;
    
    // Only 5 non-zero features in a 10000-dimensional space
    sparse_vec.push_back({100, 1.0});
    sparse_vec.push_back({500, 2.0});
    sparse_vec.push_back({1000, 3.0});
    sparse_vec.push_back({5000, 4.0});
    sparse_vec.push_back({9999, 5.0});
    sparse_vec.push_back({-1, 0.0});  // terminator
    
    EXPECT_EQ(sparse_vec.size(), 6);
    EXPECT_EQ(sparse_vec[4].index, 9999);
    EXPECT_DOUBLE_EQ(sparse_vec[4].value, 5.0);
}

// Test svm_node ordering (indices should be in ascending order)
TEST_F(SvmNodeTest, IndexOrdering) {
    std::vector<svm_node> nodes = {
        {1, 0.1},
        {3, 0.3},
        {7, 0.7},
        {10, 1.0},
        {-1, 0.0}
    };
    
    // Verify ascending order
    for (size_t i = 0; i < nodes.size() - 2; ++i) {
        EXPECT_LT(nodes[i].index, nodes[i + 1].index);
    }
}

// Test svm_node with floating point precision
TEST_F(SvmNodeTest, FloatingPointPrecision) {
    svm_node node;
    node.index = 1;
    node.value = 0.1 + 0.2;  // Known floating point issue
    
    EXPECT_NEAR(node.value, 0.3, 1e-10);
}
