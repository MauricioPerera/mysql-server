// Vector distance operations with optional AVX2+FMA SIMD optimization.
// When VECTOR_OPS_AVX2 is defined (via -DWITH_VECTOR_AVX2=ON), uses
// AVX2 intrinsics to process 8 floats per iteration.

#include "vector-common/vector_operations.h"
#include <cmath>

#ifdef VECTOR_OPS_AVX2
#include <immintrin.h>

// Horizontal sum of 8 packed floats -> single float
static inline float hsum_avx(__m256 v) {
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
}
#endif

namespace vector_operations {

double l2_distance(const float *v1, const float *v2, uint32_t dimensions) {
#ifdef VECTOR_OPS_AVX2
  __m256 sum_vec = _mm256_setzero_ps();
  uint32_t i = 0;
  for (; i + 8 <= dimensions; i += 8) {
    __m256 a = _mm256_loadu_ps(v1 + i);
    __m256 b = _mm256_loadu_ps(v2 + i);
    __m256 diff = _mm256_sub_ps(a, b);
    sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
  }
  float sum = hsum_avx(sum_vec);
  // Scalar tail for remaining elements
  for (; i < dimensions; i++) {
    float diff = v1[i] - v2[i];
    sum += diff * diff;
  }
  return std::sqrt(static_cast<double>(sum));
#else
  double sum = 0.0;
  for (uint32_t i = 0; i < dimensions; i++) {
    double diff = static_cast<double>(v1[i]) - static_cast<double>(v2[i]);
    sum += diff * diff;
  }
  return std::sqrt(sum);
#endif
}

double dot_product(const float *v1, const float *v2, uint32_t dimensions) {
#ifdef VECTOR_OPS_AVX2
  __m256 sum_vec = _mm256_setzero_ps();
  uint32_t i = 0;
  for (; i + 8 <= dimensions; i += 8) {
    __m256 a = _mm256_loadu_ps(v1 + i);
    __m256 b = _mm256_loadu_ps(v2 + i);
    sum_vec = _mm256_fmadd_ps(a, b, sum_vec);
  }
  float sum = hsum_avx(sum_vec);
  for (; i < dimensions; i++) {
    sum += v1[i] * v2[i];
  }
  return static_cast<double>(sum);
#else
  double sum = 0.0;
  for (uint32_t i = 0; i < dimensions; i++) {
    sum += static_cast<double>(v1[i]) * static_cast<double>(v2[i]);
  }
  return sum;
#endif
}

// Single-pass cosine similarity: computes dot, norm_a, norm_b simultaneously
double cosine_similarity(const float *v1, const float *v2, uint32_t dimensions) {
#ifdef VECTOR_OPS_AVX2
  __m256 dot_vec = _mm256_setzero_ps();
  __m256 norm_a_vec = _mm256_setzero_ps();
  __m256 norm_b_vec = _mm256_setzero_ps();
  uint32_t i = 0;
  for (; i + 8 <= dimensions; i += 8) {
    __m256 a = _mm256_loadu_ps(v1 + i);
    __m256 b = _mm256_loadu_ps(v2 + i);
    dot_vec = _mm256_fmadd_ps(a, b, dot_vec);
    norm_a_vec = _mm256_fmadd_ps(a, a, norm_a_vec);
    norm_b_vec = _mm256_fmadd_ps(b, b, norm_b_vec);
  }
  float dot = hsum_avx(dot_vec);
  float mag1 = hsum_avx(norm_a_vec);
  float mag2 = hsum_avx(norm_b_vec);
  // Scalar tail
  for (; i < dimensions; i++) {
    dot += v1[i] * v2[i];
    mag1 += v1[i] * v1[i];
    mag2 += v2[i] * v2[i];
  }
  double magnitude = std::sqrt(static_cast<double>(mag1)) *
                     std::sqrt(static_cast<double>(mag2));
  if (magnitude < 1e-10) return 0.0;
  return static_cast<double>(dot) / magnitude;
#else
  double dot = 0.0;
  double mag1 = 0.0;
  double mag2 = 0.0;
  for (uint32_t i = 0; i < dimensions; i++) {
    double a = static_cast<double>(v1[i]);
    double b = static_cast<double>(v2[i]);
    dot += a * b;
    mag1 += a * a;
    mag2 += b * b;
  }
  double magnitude = std::sqrt(mag1) * std::sqrt(mag2);
  if (magnitude < 1e-10) return 0.0;
  return dot / magnitude;
#endif
}

double cosine_distance(const float *v1, const float *v2, uint32_t dimensions) {
  return 1.0 - cosine_similarity(v1, v2, dimensions);
}

}  // namespace vector_operations
