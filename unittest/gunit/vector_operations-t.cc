// Fixed unit test with correct namespace comment (Issue #6)
// Changed "arrow_operations_unittest" -> "vector_operations_unittest"

#include <gtest/gtest.h>
#include "vector-common/vector_operations.h"
#include <cmath>

namespace vector_operations_unittest {

TEST(VectorOperations, L2Distance) {
  float v1[] = {0.0f, 0.0f, 0.0f};
  float v2[] = {3.0f, 4.0f, 0.0f};
  
  double result = vector_operations::l2_distance(v1, v2, 3);
  EXPECT_DOUBLE_EQ(5.0, result);
}

TEST(VectorOperations, CosineDistance) {
  float v1[] = {1.0f, 0.0f, 0.0f};
  float v2[] = {0.0f, 1.0f, 0.0f};
  
  // Orthogonal vectors: distance should be 1.0
  double result = vector_operations::cosine_distance(v1, v2, 3);
  EXPECT_NEAR(1.0, result, 1e-6);
  
  float v3[] = {1.0f, 2.0f, 3.0f};
  // Same vector: distance should be 0.0
  double result_same = vector_operations::cosine_distance(v3, v3, 3);
  EXPECT_NEAR(0.0, result_same, 1e-6);
}

TEST(VectorOperations, DotProduct) {
  float v1[] = {1.0f, 2.0f, 3.0f};
  float v2[] = {4.0f, 5.0f, 6.0f};
  
  double result = vector_operations::dot_product(v1, v2, 3);
  EXPECT_DOUBLE_EQ(32.0, result);
}

TEST(VectorOperations, CosineSimilarity) {
  float v1[] = {1.0f, 0.0f, 0.0f};
  float v2[] = {1.0f, 0.0f, 0.0f};
  
  // Same direction: similarity should be 1.0
  double result = vector_operations::cosine_similarity(v1, v2, 3);
  EXPECT_NEAR(1.0, result, 1e-6);
}

// ============================================================================
// L2 Distance edge cases
// ============================================================================

TEST(VectorOperations, L2DistanceIdenticalVectors) {
  float v[] = {1.0f, 2.0f, 3.0f};
  double result = vector_operations::l2_distance(v, v, 3);
  EXPECT_DOUBLE_EQ(0.0, result);
}

TEST(VectorOperations, L2DistanceHighDimensional) {
  // 128-dimensional unit vector along first axis vs origin-like
  const uint32_t dims = 128;
  float v1[dims];
  float v2[dims];
  for (uint32_t i = 0; i < dims; ++i) {
    v1[i] = 1.0f;  // all ones
    v2[i] = 0.0f;  // all zeros
  }
  // Distance should be sqrt(128) = 11.3137...
  double result = vector_operations::l2_distance(v1, v2, dims);
  EXPECT_NEAR(std::sqrt(128.0), result, 1e-6);
}

// ============================================================================
// Cosine edge cases
// ============================================================================

TEST(VectorOperations, CosineDistanceOppositeVectors) {
  float v1[] = {1.0f, 0.0f, 0.0f};
  float v2[] = {-1.0f, 0.0f, 0.0f};

  // Opposite direction: similarity = -1, distance = 1 - (-1) = 2.0
  double result = vector_operations::cosine_distance(v1, v2, 3);
  EXPECT_NEAR(2.0, result, 1e-6);
}

TEST(VectorOperations, CosineDistanceZeroVector) {
  float v1[] = {0.0f, 0.0f, 0.0f};
  float v2[] = {1.0f, 2.0f, 3.0f};

  // Zero vector: magnitude is 0, should return 1.0 (cosine_similarity returns 0)
  double result = vector_operations::cosine_distance(v1, v2, 3);
  EXPECT_NEAR(1.0, result, 1e-6);
}

// ============================================================================
// Dot product edge cases
// ============================================================================

TEST(VectorOperations, DotProductNegative) {
  float v1[] = {-1.0f, -2.0f, -3.0f};
  float v2[] = {4.0f, 5.0f, 6.0f};

  // (-1*4) + (-2*5) + (-3*6) = -4 -10 -18 = -32
  double result = vector_operations::dot_product(v1, v2, 3);
  EXPECT_DOUBLE_EQ(-32.0, result);
}

TEST(VectorOperations, DotProductOrthogonal) {
  float v1[] = {1.0f, 0.0f, 0.0f};
  float v2[] = {0.0f, 1.0f, 0.0f};

  double result = vector_operations::dot_product(v1, v2, 3);
  EXPECT_DOUBLE_EQ(0.0, result);
}

}  // namespace vector_operations_unittest
