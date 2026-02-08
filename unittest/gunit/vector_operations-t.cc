// Unit tests for vector distance operations (scalar and SIMD paths)

#include <gtest/gtest.h>
#include "vector-common/vector_operations.h"
#include <cmath>
#include <vector>

namespace vector_operations_unittest {

TEST(VectorOperations, L2Distance) {
  float v1[] = {0.0f, 0.0f, 0.0f};
  float v2[] = {3.0f, 4.0f, 0.0f};

  double result = vector_operations::l2_distance(v1, v2, 3);
  EXPECT_NEAR(5.0, result, 1e-5);
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
  EXPECT_NEAR(32.0, result, 1e-4);
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
  EXPECT_NEAR(0.0, result, 1e-10);
}

TEST(VectorOperations, L2DistanceHighDimensional) {
  // 128-dimensional: all ones vs all zeros
  const uint32_t dims = 128;
  float v1[dims];
  float v2[dims];
  for (uint32_t i = 0; i < dims; ++i) {
    v1[i] = 1.0f;
    v2[i] = 0.0f;
  }
  // Distance should be sqrt(128) = 11.3137...
  double result = vector_operations::l2_distance(v1, v2, dims);
  EXPECT_NEAR(std::sqrt(128.0), result, 1e-4);
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
  EXPECT_NEAR(-32.0, result, 1e-4);
}

TEST(VectorOperations, DotProductOrthogonal) {
  float v1[] = {1.0f, 0.0f, 0.0f};
  float v2[] = {0.0f, 1.0f, 0.0f};

  double result = vector_operations::dot_product(v1, v2, 3);
  EXPECT_NEAR(0.0, result, 1e-10);
}

// ============================================================================
// SIMD tail-loop coverage (non-multiple-of-8 dimensions)
// ============================================================================

TEST(VectorOperations, L2DistanceNonAligned) {
  // 7 dimensions: not divisible by 4 or 8
  float v1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  float v2[] = {7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  double result = vector_operations::l2_distance(v1, v2, 7);
  // diffs: 6,4,2,0,2,4,6 -> sum of squares: 36+16+4+0+4+16+36 = 112
  EXPECT_NEAR(std::sqrt(112.0), result, 1e-4);
}

TEST(VectorOperations, DotProductNonAligned) {
  float v1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  float v2[] = {7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  double result = vector_operations::dot_product(v1, v2, 7);
  // 7+12+15+16+15+12+7 = 84
  EXPECT_NEAR(84.0, result, 1e-4);
}

TEST(VectorOperations, CosineSimLargeVector) {
  // 1536 dimensions (OpenAI ada-002 embedding size)
  const uint32_t dims = 1536;
  std::vector<float> v1(dims), v2(dims);
  for (uint32_t i = 0; i < dims; ++i) {
    v1[i] = static_cast<float>(i) / static_cast<float>(dims);
    v2[i] = static_cast<float>(dims - i) / static_cast<float>(dims);
  }
  double result = vector_operations::cosine_similarity(v1.data(), v2.data(), dims);
  // Verify result is in valid range [-1, 1]
  EXPECT_GE(result, -1.0);
  EXPECT_LE(result, 1.0);
}

// ============================================================================
// std::vector overloads
// ============================================================================

TEST(VectorOperations, VectorOverloadL2) {
  std::vector<float> v1 = {0.0f, 0.0f, 0.0f};
  std::vector<float> v2 = {3.0f, 4.0f, 0.0f};
  double result = vector_operations::l2_distance(v1, v2);
  EXPECT_NEAR(5.0, result, 1e-5);
}

TEST(VectorOperations, NegDotProduct) {
  std::vector<float> v1 = {1.0f, 2.0f, 3.0f};
  std::vector<float> v2 = {4.0f, 5.0f, 6.0f};
  double result = vector_operations::neg_dot_product(v1, v2);
  EXPECT_NEAR(-32.0, result, 1e-4);
}

}  // namespace vector_operations_unittest
