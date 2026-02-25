/**
  @file unittest/gunit/vector_benchmark-t.cc

  Performance benchmarks for vector distance operations and HNSW search.
  Each test prints timing results (ns/op) for visibility.
  Tests always PASS - they are informational.
*/

#include <gtest/gtest.h>
#include "storage/innobase/vector/vec0hnsw.h"
#include "vector-common/vector_operations.h"
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

namespace vector_benchmark_unittest {

using innodb_vector::HnswIndex;
using innodb_vector::hnsw_config_t;
using innodb_vector::hnsw_metric_t;

static std::vector<float> random_vector(uint32_t dims, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(dims);
  for (uint32_t i = 0; i < dims; ++i) v[i] = dist(rng);
  return v;
}

// Prevent compiler from optimizing away the result
static volatile double benchmark_sink;

// ----------------------------------------------------------------------------
// Benchmark 1: L2 Distance
// ----------------------------------------------------------------------------
TEST(VectorBenchmark, L2Distance) {
  const int iterations = 100000;

  for (uint32_t dims : {128u, 1536u}) {
    auto v1 = random_vector(dims, 1);
    auto v2 = random_vector(dims, 2);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
      benchmark_sink = vector_operations::l2_distance(
          v1.data(), v2.data(), dims);
    }
    auto end = std::chrono::steady_clock::now();

    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  end - start).count();
    printf("  L2 distance  %4u dims: %6lld ns/op  (%d iterations)\n",
           dims, static_cast<long long>(ns / iterations), iterations);
  }
}

// ----------------------------------------------------------------------------
// Benchmark 2: Dot Product
// ----------------------------------------------------------------------------
TEST(VectorBenchmark, DotProduct) {
  const int iterations = 100000;

  for (uint32_t dims : {128u, 1536u}) {
    auto v1 = random_vector(dims, 3);
    auto v2 = random_vector(dims, 4);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
      benchmark_sink = vector_operations::dot_product(
          v1.data(), v2.data(), dims);
    }
    auto end = std::chrono::steady_clock::now();

    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  end - start).count();
    printf("  Dot product  %4u dims: %6lld ns/op  (%d iterations)\n",
           dims, static_cast<long long>(ns / iterations), iterations);
  }
}

// ----------------------------------------------------------------------------
// Benchmark 3: Cosine Similarity
// ----------------------------------------------------------------------------
TEST(VectorBenchmark, CosineSimilarity) {
  const int iterations = 100000;

  for (uint32_t dims : {128u, 1536u}) {
    auto v1 = random_vector(dims, 5);
    auto v2 = random_vector(dims, 6);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
      benchmark_sink = vector_operations::cosine_similarity(
          v1.data(), v2.data(), dims);
    }
    auto end = std::chrono::steady_clock::now();

    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  end - start).count();
    printf("  Cosine sim   %4u dims: %6lld ns/op  (%d iterations)\n",
           dims, static_cast<long long>(ns / iterations), iterations);
  }
}

// ----------------------------------------------------------------------------
// Benchmark 4: HNSW Search latency
// ----------------------------------------------------------------------------
TEST(VectorBenchmark, HnswSearch) {
  hnsw_config_t config;
  config.M = 16;
  config.M0 = 32;
  config.ef_construction = 100;
  config.ef_search = 50;
  config.max_elements = 5000;
  config.dimensions = 128;
  config.metric = hnsw_metric_t::L2;

  HnswIndex index(config);

  // Build index with 1000 vectors
  for (int i = 0; i < 1000; ++i) {
    auto v = random_vector(config.dimensions, static_cast<unsigned>(i));
    index.insert(static_cast<uint64_t>(i), v);
  }

  const int searches = 1000;
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < searches; ++i) {
    auto q = random_vector(config.dimensions,
                            static_cast<unsigned>(10000 + i));
    auto results = index.search(q, 10);
    benchmark_sink = results.empty() ? 0.0 : results[0].distance;
  }
  auto end = std::chrono::steady_clock::now();

  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                end - start).count();
  printf("  HNSW search (k=10, 1000 vectors, 128d): %lld us/query  (%d queries)\n",
         static_cast<long long>(us / searches), searches);
}

}  // namespace vector_benchmark_unittest
