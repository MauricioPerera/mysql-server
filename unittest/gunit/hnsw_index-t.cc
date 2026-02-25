/**
  @file unittest/gunit/hnsw_index-t.cc

  Unit tests for HNSW Index implementation
*/

#include <gtest/gtest.h>
#include "storage/innobase/vector/vec0hnsw.h"
#include <random>
#include <cmath>
#include <cstdio>
#include <string>

namespace innodb_vector_unittest {

class HnswIndexTest : public ::testing::Test {
 protected:
  void SetUp() override {
    config_.M = 16;
    config_.M0 = 32;
    config_.ef_construction = 100;
    config_.ef_search = 50;
    config_.max_elements = 10000;
    config_.dimensions = 128;
  }

  innodb_vector::hnsw_config_t config_;
  
  std::vector<float> random_vector(size_t dims) {
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vec(dims);
    for (auto &v : vec) v = dist(rng);
    return vec;
  }

  double l2_distance(const std::vector<float> &a, const std::vector<float> &b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      double diff = a[i] - b[i];
      sum += diff * diff;
    }
    return std::sqrt(sum);
  }
};

TEST_F(HnswIndexTest, EmptyIndex) {
  innodb_vector::HnswIndex index(config_);
  EXPECT_EQ(0u, index.size());
  
  auto results = index.search(random_vector(128), 10);
  EXPECT_TRUE(results.empty());
}

TEST_F(HnswIndexTest, SingleInsert) {
  innodb_vector::HnswIndex index(config_);
  auto vec = random_vector(128);
  
  EXPECT_TRUE(index.insert(1, vec));
  EXPECT_EQ(1u, index.size());
}

TEST_F(HnswIndexTest, MultipleInserts) {
  innodb_vector::HnswIndex index(config_);
  
  for (uint64_t i = 0; i < 100; ++i) {
    EXPECT_TRUE(index.insert(i, random_vector(128)));
  }
  
  EXPECT_EQ(100u, index.size());
}

TEST_F(HnswIndexTest, ExternalIdCorrectness) {
  innodb_vector::HnswIndex index(config_);
  
  // Insert with non-sequential, large IDs to ensure we aren't returning internal indices (0, 1, 2...)
  uint64_t id1 = 1001;
  uint64_t id2 = 5005;
  uint64_t id3 = 9999;
  
  auto vec1 = random_vector(128);
  auto vec2 = random_vector(128);
  auto vec3 = random_vector(128);
  
  index.insert(id1, vec1);
  index.insert(id2, vec2);
  index.insert(id3, vec3);
  
  auto results = index.search(vec2, 1);
  ASSERT_FALSE(results.empty());
  EXPECT_EQ(id2, results[0].id) << "Should return external ID " << id2 << ", not internal index";
  
  results = index.search(vec3, 1);
  ASSERT_FALSE(results.empty());
  EXPECT_EQ(id3, results[0].id) << "Should return external ID " << id3;
}

TEST_F(HnswIndexTest, SearchFindsExactMatch) {
  innodb_vector::HnswIndex index(config_);
  
  // Insert vectors with known IDs using a stride/offset
  std::vector<std::vector<float>> vectors;
  uint64_t id_offset = 10000;
  
  for (uint64_t i = 0; i < 100; ++i) {
    auto vec = random_vector(128);
    vectors.push_back(vec);
    index.insert(id_offset + i, vec);
  }
  
  // Search for vector 50 (ID 10050)
  auto results = index.search(vectors[50], 1);
  
  ASSERT_FALSE(results.empty());
  EXPECT_EQ(id_offset + 50, results[0].id);
  EXPECT_NEAR(0.0, results[0].distance, 1e-6);
}

TEST_F(HnswIndexTest, SearchReturnsKNearest) {
  innodb_vector::HnswIndex index(config_);
  
  for (uint64_t i = 0; i < 1000; ++i) {
    // ID = i * 2 to distinguish from index
    index.insert(i * 2, random_vector(128));
  }
  
  auto query = random_vector(128);
  auto results = index.search(query, 10);
  
  EXPECT_EQ(10u, results.size());
  
  // Verify results are sorted by distance
  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_LE(results[i-1].distance, results[i].distance);
    // Verify ID is even (simple check that we got our IDs back)
    EXPECT_EQ(0u, results[i].id % 2);
  }
}

TEST_F(HnswIndexTest, RecallQuality) {
  // Simple recall test - exact search should have high recall
  innodb_vector::HnswIndex index(config_);
  
  std::vector<std::vector<float>> vectors;
  for (uint64_t i = 0; i < 500; ++i) {
    auto vec = random_vector(128);
    vectors.push_back(vec);
    index.insert(i + 100, vec); // ID offset
  }
  
  // For each random query, check if top-1 is reasonable
  int correct = 0;
  for (int trial = 0; trial < 100; ++trial) {
    uint64_t target_idx = trial % 500;
    uint64_t target_id = target_idx + 100;
    
    auto results = index.search(vectors[target_idx], 1);
    
    if (!results.empty() && results[0].id == target_id) {
      ++correct;
    }
  }
  
  // Expect at least 95% recall for exact matches
  EXPECT_GE(correct, 95);
}

// ============================================================================
// Remove operations
// ============================================================================

TEST_F(HnswIndexTest, RemoveExisting) {
  innodb_vector::HnswIndex index(config_);

  auto vec = random_vector(128);
  index.insert(1, vec);
  EXPECT_EQ(1u, index.size());

  EXPECT_TRUE(index.remove(1));
  EXPECT_EQ(0u, index.size());
  EXPECT_EQ(1u, index.deleted_count());
}

TEST_F(HnswIndexTest, RemoveNonExistent) {
  innodb_vector::HnswIndex index(config_);

  index.insert(1, random_vector(128));
  EXPECT_FALSE(index.remove(999));
  EXPECT_EQ(1u, index.size());
}

TEST_F(HnswIndexTest, RemoveAndSearch) {
  innodb_vector::HnswIndex index(config_);

  auto vec1 = random_vector(128);
  auto vec2 = random_vector(128);
  auto vec3 = random_vector(128);

  index.insert(10, vec1);
  index.insert(20, vec2);
  index.insert(30, vec3);

  // Remove vec2
  EXPECT_TRUE(index.remove(20));

  // Search for vec2 - should NOT find id=20 in results
  auto results = index.search(vec2, 3);
  for (const auto &r : results) {
    EXPECT_NE(20u, r.id) << "Deleted vector should not appear in results";
  }
}

TEST_F(HnswIndexTest, RemoveEntryPoint) {
  innodb_vector::HnswIndex index(config_);

  // Insert several vectors
  for (uint64_t i = 0; i < 20; ++i) {
    index.insert(i + 1, random_vector(128));
  }

  // Remove the first inserted element (likely the entry point)
  EXPECT_TRUE(index.remove(1));
  EXPECT_EQ(19u, index.size());

  // Index should still be searchable
  auto results = index.search(random_vector(128), 5);
  EXPECT_FALSE(results.empty());
}

TEST_F(HnswIndexTest, RemoveAll) {
  innodb_vector::HnswIndex index(config_);

  for (uint64_t i = 0; i < 10; ++i) {
    index.insert(i + 1, random_vector(128));
  }

  for (uint64_t i = 0; i < 10; ++i) {
    EXPECT_TRUE(index.remove(i + 1));
  }

  EXPECT_EQ(0u, index.size());
  auto results = index.search(random_vector(128), 5);
  EXPECT_TRUE(results.empty());
}

// ============================================================================
// Update operations
// ============================================================================

TEST_F(HnswIndexTest, UpdateExisting) {
  innodb_vector::HnswIndex index(config_);

  auto original = random_vector(128);
  auto updated = random_vector(128);
  index.insert(42, original);

  EXPECT_TRUE(index.update(42, updated));
  EXPECT_EQ(1u, index.size());

  // Search for the updated vector should find id=42
  auto results = index.search(updated, 1);
  ASSERT_FALSE(results.empty());
  EXPECT_EQ(42u, results[0].id);
  EXPECT_NEAR(0.0, results[0].distance, 1e-6);
}

TEST_F(HnswIndexTest, UpdateDimensionMismatch) {
  innodb_vector::HnswIndex index(config_);

  index.insert(1, random_vector(128));

  // Update with wrong dimensions should fail
  auto wrong_dims = random_vector(64);
  EXPECT_FALSE(index.update(1, wrong_dims));
}

// ============================================================================
// Contains
// ============================================================================

TEST_F(HnswIndexTest, ContainsExisting) {
  innodb_vector::HnswIndex index(config_);

  index.insert(100, random_vector(128));
  index.insert(200, random_vector(128));

  EXPECT_TRUE(index.contains(100));
  EXPECT_TRUE(index.contains(200));
  EXPECT_FALSE(index.contains(300));
}

TEST_F(HnswIndexTest, ContainsAfterRemove) {
  innodb_vector::HnswIndex index(config_);

  index.insert(100, random_vector(128));
  EXPECT_TRUE(index.contains(100));

  index.remove(100);
  EXPECT_FALSE(index.contains(100));
}

// ============================================================================
// Duplicate ID rejection
// ============================================================================

TEST_F(HnswIndexTest, DuplicateIdRejected) {
  innodb_vector::HnswIndex index(config_);

  EXPECT_TRUE(index.insert(1, random_vector(128)));
  EXPECT_FALSE(index.insert(1, random_vector(128)));
  EXPECT_EQ(1u, index.size());
}

// ============================================================================
// Free list reuse after remove
// ============================================================================

TEST_F(HnswIndexTest, FreeListReuse) {
  innodb_vector::HnswIndex index(config_);

  for (uint64_t i = 0; i < 5; ++i) {
    index.insert(i + 1, random_vector(128));
  }
  uint64_t total_before = index.total_nodes();

  // Remove some
  index.remove(2);
  index.remove(4);

  // Insert new elements - should reuse freed slots
  index.insert(100, random_vector(128));
  index.insert(200, random_vector(128));

  // Total node slots should not have grown
  EXPECT_EQ(total_before, index.total_nodes());
  EXPECT_EQ(5u, index.size());
}

// ============================================================================
// Persistence (save/load)
// ============================================================================

TEST_F(HnswIndexTest, SaveAndLoad) {
  innodb_vector::HnswIndex index(config_);

  std::vector<std::vector<float>> vectors;
  for (uint64_t i = 0; i < 50; ++i) {
    auto vec = random_vector(128);
    vectors.push_back(vec);
    index.insert(i + 1000, vec);
  }

  // Save to temp file
  std::string temp_path = std::tmpnam(nullptr);
  temp_path += ".hnsw";
  ASSERT_TRUE(index.save_to_file(temp_path.c_str()));

  // Load into a new index
  innodb_vector::HnswIndex loaded(config_);
  ASSERT_TRUE(loaded.load_from_file(temp_path.c_str()));

  EXPECT_EQ(index.size(), loaded.size());

  // Verify search works on loaded index
  auto results = loaded.search(vectors[25], 1);
  ASSERT_FALSE(results.empty());
  EXPECT_EQ(1025u, results[0].id);
  EXPECT_NEAR(0.0, results[0].distance, 1e-6);

  // Cleanup
  std::remove(temp_path.c_str());
}

TEST_F(HnswIndexTest, LoadInvalidPath) {
  innodb_vector::HnswIndex index(config_);
  EXPECT_FALSE(index.load_from_file("/nonexistent/path/index.hnsw"));
}

TEST_F(HnswIndexTest, SaveLoadPreservesMetric) {
  config_.metric = innodb_vector::hnsw_metric_t::COSINE;
  config_.dimensions = 3;
  innodb_vector::HnswIndex index(config_);

  index.insert(1, {1.0f, 0.0f, 0.0f});
  index.insert(2, {0.0f, 1.0f, 0.0f});

  std::string temp_path = std::tmpnam(nullptr);
  temp_path += ".hnsw";
  ASSERT_TRUE(index.save_to_file(temp_path.c_str()));

  innodb_vector::hnsw_config_t empty_config;
  innodb_vector::HnswIndex loaded(empty_config);
  ASSERT_TRUE(loaded.load_from_file(temp_path.c_str()));

  // The loaded config should have COSINE metric
  EXPECT_EQ(innodb_vector::hnsw_metric_t::COSINE, loaded.config().metric);
  EXPECT_EQ(3u, loaded.config().dimensions);

  std::remove(temp_path.c_str());
}

// ============================================================================
// Distance metrics
// ============================================================================

TEST_F(HnswIndexTest, SearchWithCosineMetric) {
  config_.metric = innodb_vector::hnsw_metric_t::COSINE;
  config_.dimensions = 3;
  innodb_vector::HnswIndex index(config_);

  // Insert vectors in known directions
  index.insert(1, {1.0f, 0.0f, 0.0f});   // x-axis
  index.insert(2, {0.0f, 1.0f, 0.0f});   // y-axis
  index.insert(3, {0.7071f, 0.7071f, 0.0f}); // 45 degrees

  // Query along x-axis - closest should be id=1 (same direction)
  auto results = index.search({1.0f, 0.0f, 0.0f}, 3);
  ASSERT_GE(results.size(), 1u);
  EXPECT_EQ(1u, results[0].id);
  EXPECT_NEAR(0.0, results[0].distance, 1e-4);
}

TEST_F(HnswIndexTest, SearchWithDotProductMetric) {
  config_.metric = innodb_vector::hnsw_metric_t::DOT_PRODUCT;
  config_.dimensions = 3;
  innodb_vector::HnswIndex index(config_);

  index.insert(1, {1.0f, 0.0f, 0.0f});
  index.insert(2, {5.0f, 0.0f, 0.0f});   // larger magnitude
  index.insert(3, {0.0f, 1.0f, 0.0f});

  // Query along x-axis - highest dot product with id=2 (5.0)
  auto results = index.search({1.0f, 0.0f, 0.0f}, 3);
  ASSERT_GE(results.size(), 1u);
  // Dot product uses negative values, so most similar = most negative distance
  EXPECT_EQ(2u, results[0].id);
  EXPECT_NEAR(-5.0, results[0].distance, 1e-4);
}

TEST_F(HnswIndexTest, DistanceL2Correctness) {
  config_.dimensions = 3;
  innodb_vector::HnswIndex index(config_);

  index.insert(1, {0.0f, 0.0f, 0.0f});
  index.insert(2, {3.0f, 4.0f, 0.0f});

  auto results = index.search({0.0f, 0.0f, 0.0f}, 2);
  ASSERT_EQ(2u, results.size());
  EXPECT_EQ(1u, results[0].id);
  EXPECT_NEAR(0.0, results[0].distance, 1e-6);
  EXPECT_EQ(2u, results[1].id);
  EXPECT_NEAR(5.0, results[1].distance, 1e-4);  // sqrt(9+16) = 5
}

TEST_F(HnswIndexTest, DistanceCosineCorrectness) {
  config_.metric = innodb_vector::hnsw_metric_t::COSINE;
  config_.dimensions = 3;
  innodb_vector::HnswIndex index(config_);

  index.insert(1, {1.0f, 0.0f, 0.0f});
  index.insert(2, {0.0f, 1.0f, 0.0f});  // orthogonal

  // Search with x-axis vector
  auto results = index.search({1.0f, 0.0f, 0.0f}, 2);
  ASSERT_EQ(2u, results.size());

  // id=1 should have distance ~0 (same direction)
  EXPECT_EQ(1u, results[0].id);
  EXPECT_NEAR(0.0, results[0].distance, 1e-6);

  // id=2 should have distance ~1.0 (orthogonal: 1 - cos(90) = 1)
  EXPECT_EQ(2u, results[1].id);
  EXPECT_NEAR(1.0, results[1].distance, 1e-4);
}

TEST_F(HnswIndexTest, DistanceDotProductCorrectness) {
  config_.metric = innodb_vector::hnsw_metric_t::DOT_PRODUCT;
  config_.dimensions = 3;
  innodb_vector::HnswIndex index(config_);

  index.insert(1, {1.0f, 2.0f, 3.0f});
  index.insert(2, {4.0f, 5.0f, 6.0f});

  // Query with [1,1,1] -> dot with id1 = 6, dot with id2 = 15
  // Distance = -dot, so id2 has distance -15 (lower = better)
  auto results = index.search({1.0f, 1.0f, 1.0f}, 2);
  ASSERT_EQ(2u, results.size());
  EXPECT_EQ(2u, results[0].id);
  EXPECT_NEAR(-15.0, results[0].distance, 1e-4);
  EXPECT_EQ(1u, results[1].id);
  EXPECT_NEAR(-6.0, results[1].distance, 1e-4);
}

// ============================================================================
// Search edge cases
// ============================================================================

TEST_F(HnswIndexTest, SearchKGreaterThanSize) {
  innodb_vector::HnswIndex index(config_);

  for (uint64_t i = 0; i < 5; ++i) {
    index.insert(i + 1, random_vector(128));
  }

  // Ask for more results than elements
  auto results = index.search(random_vector(128), 100);
  EXPECT_EQ(5u, results.size());
}

// ============================================================================
// Remove + re-insert same ID
// ============================================================================

TEST_F(HnswIndexTest, RemoveAndReinsert) {
  innodb_vector::HnswIndex index(config_);

  auto vec1 = random_vector(128);
  auto vec2 = random_vector(128);

  index.insert(42, vec1);
  EXPECT_TRUE(index.remove(42));
  EXPECT_FALSE(index.contains(42));

  // Re-insert with same ID, different vector
  EXPECT_TRUE(index.insert(42, vec2));
  EXPECT_TRUE(index.contains(42));

  // Search should find the new vector
  auto results = index.search(vec2, 1);
  ASSERT_FALSE(results.empty());
  EXPECT_EQ(42u, results[0].id);
  EXPECT_NEAR(0.0, results[0].distance, 1e-6);
}

}  // namespace innodb_vector_unittest
