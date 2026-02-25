// Copyright (c) 2025, Oracle and/or its affiliates.

#ifndef VECTOR_OPERATIONS_INCLUDED
#define VECTOR_OPERATIONS_INCLUDED

#include <algorithm>
#include <cstdint>
#include <vector>

namespace vector_operations {

// Core API (float pointer)
double l2_distance(const float *v1, const float *v2, uint32_t dimensions);
double cosine_similarity(const float *v1, const float *v2, uint32_t dimensions);
double cosine_distance(const float *v1, const float *v2, uint32_t dimensions);
double dot_product(const float *v1, const float *v2, uint32_t dimensions);

// std::vector convenience overloads (delegate to pointer API)
inline double l2_distance(const std::vector<float> &v1,
                           const std::vector<float> &v2) {
  auto n = static_cast<uint32_t>(std::min(v1.size(), v2.size()));
  return l2_distance(v1.data(), v2.data(), n);
}

inline double cosine_similarity(const std::vector<float> &v1,
                                 const std::vector<float> &v2) {
  auto n = static_cast<uint32_t>(std::min(v1.size(), v2.size()));
  return cosine_similarity(v1.data(), v2.data(), n);
}

inline double cosine_distance(const std::vector<float> &v1,
                               const std::vector<float> &v2) {
  auto n = static_cast<uint32_t>(std::min(v1.size(), v2.size()));
  return cosine_distance(v1.data(), v2.data(), n);
}

inline double dot_product(const std::vector<float> &v1,
                           const std::vector<float> &v2) {
  auto n = static_cast<uint32_t>(std::min(v1.size(), v2.size()));
  return dot_product(v1.data(), v2.data(), n);
}

// Negative dot product for HNSW MIPS (higher dot = smaller distance)
inline double neg_dot_product(const std::vector<float> &v1,
                               const std::vector<float> &v2) {
  return -dot_product(v1, v2);
}

}  // namespace vector_operations

#endif  // VECTOR_OPERATIONS_INCLUDED
