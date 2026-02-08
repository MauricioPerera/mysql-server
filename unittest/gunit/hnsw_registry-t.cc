/**
  @file unittest/gunit/hnsw_registry-t.cc

  Unit tests for HNSW Index Registry
*/

#include <gtest/gtest.h>
#include "storage/innobase/include/vec0hnsw_registry.h"
#include <algorithm>
#include <string>
#include <vector>

namespace innodb_vector_unittest {

class HnswRegistryTest : public ::testing::Test {
 protected:
  void TearDown() override {
    // Clean up all indexes created during tests
    auto &registry = innodb_vector::HnswIndexRegistry::instance();
    auto keys = registry.list_indexes();
    for (const auto &key : keys) {
      // Parse key into table:column
      auto colon = key.find(':');
      if (colon != std::string::npos) {
        registry.drop_index(key.substr(0, colon), key.substr(colon + 1));
      } else {
        registry.drop_index(key);
      }
    }
  }

  innodb_vector::HnswIndexRegistry &registry() {
    return innodb_vector::HnswIndexRegistry::instance();
  }
};

TEST_F(HnswRegistryTest, RegisterAndGet) {
  EXPECT_TRUE(registry().register_index("test_table", "embedding", 128));

  auto *idx = registry().get_index("test_table", "embedding");
  ASSERT_NE(nullptr, idx);
  EXPECT_EQ(128u, idx->config().dimensions);
}

TEST_F(HnswRegistryTest, RegisterDuplicate) {
  EXPECT_TRUE(registry().register_index("t1", "col1", 64));
  EXPECT_FALSE(registry().register_index("t1", "col1", 64));
}

TEST_F(HnswRegistryTest, GetNonExistent) {
  EXPECT_EQ(nullptr, registry().get_index("no_such_table", "no_col"));
}

TEST_F(HnswRegistryTest, DropExisting) {
  registry().register_index("drop_test", "vec", 32);
  EXPECT_TRUE(registry().has_index("drop_test", "vec"));

  EXPECT_TRUE(registry().drop_index("drop_test", "vec"));
  EXPECT_FALSE(registry().has_index("drop_test", "vec"));
  EXPECT_EQ(nullptr, registry().get_index("drop_test", "vec"));
}

TEST_F(HnswRegistryTest, DropNonExistent) {
  EXPECT_FALSE(registry().drop_index("nonexistent"));
}

TEST_F(HnswRegistryTest, HasIndex) {
  registry().register_index("has_test", "v1", 64);

  EXPECT_TRUE(registry().has_index("has_test", "v1"));
  EXPECT_FALSE(registry().has_index("has_test", "v2"));
  EXPECT_FALSE(registry().has_index("other_table", "v1"));
}

TEST_F(HnswRegistryTest, ListIndexes) {
  registry().register_index("tA", "c1", 64);
  registry().register_index("tA", "c2", 128);
  registry().register_index("tB", "c1", 256);

  auto keys = registry().list_indexes();
  EXPECT_EQ(3u, keys.size());

  // Check all keys are present (order is unspecified)
  std::sort(keys.begin(), keys.end());
  EXPECT_NE(keys.end(), std::find(keys.begin(), keys.end(), "tA:c1"));
  EXPECT_NE(keys.end(), std::find(keys.begin(), keys.end(), "tA:c2"));
  EXPECT_NE(keys.end(), std::find(keys.begin(), keys.end(), "tB:c1"));
}

TEST_F(HnswRegistryTest, MultiColumnSameTable) {
  EXPECT_TRUE(registry().register_index("docs", "emb_title", 128));
  EXPECT_TRUE(registry().register_index("docs", "emb_body", 512));
  EXPECT_TRUE(registry().register_index("docs", "emb_image", 768));

  auto *idx1 = registry().get_index("docs", "emb_title");
  auto *idx2 = registry().get_index("docs", "emb_body");
  auto *idx3 = registry().get_index("docs", "emb_image");

  ASSERT_NE(nullptr, idx1);
  ASSERT_NE(nullptr, idx2);
  ASSERT_NE(nullptr, idx3);

  EXPECT_EQ(128u, idx1->config().dimensions);
  EXPECT_EQ(512u, idx2->config().dimensions);
  EXPECT_EQ(768u, idx3->config().dimensions);

  // They should be different index instances
  EXPECT_NE(idx1, idx2);
  EXPECT_NE(idx2, idx3);
}

TEST_F(HnswRegistryTest, GetColumnsForTable) {
  registry().register_index("multi", "col_a", 64);
  registry().register_index("multi", "col_b", 128);
  registry().register_index("other_table", "col_x", 256);

  auto cols = registry().get_columns_for_table("multi");
  EXPECT_EQ(2u, cols.size());

  std::sort(cols.begin(), cols.end());
  EXPECT_EQ("col_a", cols[0]);
  EXPECT_EQ("col_b", cols[1]);

  // Other table should not be included
  auto other_cols = registry().get_columns_for_table("other_table");
  EXPECT_EQ(1u, other_cols.size());
  EXPECT_EQ("col_x", other_cols[0]);

  // Non-existent table
  auto empty = registry().get_columns_for_table("nope");
  EXPECT_TRUE(empty.empty());
}

TEST_F(HnswRegistryTest, LegacyMode) {
  // Register without column name (legacy)
  EXPECT_TRUE(registry().register_index("legacy_table", 128));

  auto *idx = registry().get_index("legacy_table");
  ASSERT_NE(nullptr, idx);

  // get_columns_for_table should return empty string for legacy entry
  auto cols = registry().get_columns_for_table("legacy_table");
  EXPECT_EQ(1u, cols.size());
  EXPECT_EQ("", cols[0]);

  EXPECT_TRUE(registry().drop_index("legacy_table"));
}

// ============================================================================
// Metric parsing
// ============================================================================

TEST_F(HnswRegistryTest, ParseMetricL2) {
  using innodb_vector::hnsw_metric_t;
  using innodb_vector::HnswIndexRegistry;

  EXPECT_EQ(hnsw_metric_t::L2, HnswIndexRegistry::parse_metric("l2"));
  EXPECT_EQ(hnsw_metric_t::L2, HnswIndexRegistry::parse_metric("L2"));
  EXPECT_EQ(hnsw_metric_t::L2, HnswIndexRegistry::parse_metric("unknown"));
  EXPECT_EQ(hnsw_metric_t::L2, HnswIndexRegistry::parse_metric(""));
}

TEST_F(HnswRegistryTest, ParseMetricCosine) {
  using innodb_vector::hnsw_metric_t;
  using innodb_vector::HnswIndexRegistry;

  EXPECT_EQ(hnsw_metric_t::COSINE, HnswIndexRegistry::parse_metric("cosine"));
  EXPECT_EQ(hnsw_metric_t::COSINE, HnswIndexRegistry::parse_metric("cos"));
  EXPECT_EQ(hnsw_metric_t::COSINE, HnswIndexRegistry::parse_metric("COSINE"));
  EXPECT_EQ(hnsw_metric_t::COSINE, HnswIndexRegistry::parse_metric("Cos"));
}

TEST_F(HnswRegistryTest, ParseMetricDotProduct) {
  using innodb_vector::hnsw_metric_t;
  using innodb_vector::HnswIndexRegistry;

  EXPECT_EQ(hnsw_metric_t::DOT_PRODUCT,
            HnswIndexRegistry::parse_metric("dot_product"));
  EXPECT_EQ(hnsw_metric_t::DOT_PRODUCT,
            HnswIndexRegistry::parse_metric("dot"));
  EXPECT_EQ(hnsw_metric_t::DOT_PRODUCT,
            HnswIndexRegistry::parse_metric("ip"));
  EXPECT_EQ(hnsw_metric_t::DOT_PRODUCT,
            HnswIndexRegistry::parse_metric("inner_product"));
  EXPECT_EQ(hnsw_metric_t::DOT_PRODUCT,
            HnswIndexRegistry::parse_metric("DOT_PRODUCT"));
}

TEST_F(HnswRegistryTest, MetricToString) {
  using innodb_vector::hnsw_metric_t;
  using innodb_vector::HnswIndexRegistry;

  EXPECT_STREQ("l2", HnswIndexRegistry::metric_to_string(hnsw_metric_t::L2));
  EXPECT_STREQ("cosine",
               HnswIndexRegistry::metric_to_string(hnsw_metric_t::COSINE));
  EXPECT_STREQ(
      "dot_product",
      HnswIndexRegistry::metric_to_string(hnsw_metric_t::DOT_PRODUCT));
}

TEST_F(HnswRegistryTest, RegisterWithMetric) {
  registry().register_index("metric_test", "vec",
                            64, 16, 200,
                            innodb_vector::hnsw_metric_t::COSINE);

  auto *idx = registry().get_index("metric_test", "vec");
  ASSERT_NE(nullptr, idx);
  EXPECT_EQ(innodb_vector::hnsw_metric_t::COSINE, idx->config().metric);
}

TEST_F(HnswRegistryTest, ReregisterAfterDrop) {
  registry().register_index("reuse", "v", 64);
  EXPECT_TRUE(registry().drop_index("reuse", "v"));

  // Should be able to register again
  EXPECT_TRUE(registry().register_index("reuse", "v", 128));
  auto *idx = registry().get_index("reuse", "v");
  ASSERT_NE(nullptr, idx);
  EXPECT_EQ(128u, idx->config().dimensions);
}

}  // namespace innodb_vector_unittest
