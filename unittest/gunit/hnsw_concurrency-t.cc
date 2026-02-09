/**
  @file unittest/gunit/hnsw_concurrency-t.cc

  Concurrency tests for HnswIndex shared_mutex correctness.
  Verifies that concurrent reads (search/contains) and writes
  (insert/remove/update) do not corrupt the index or crash.
*/

#include <gtest/gtest.h>
#include "storage/innobase/vector/vec0hnsw.h"
#include <atomic>
#include <random>
#include <thread>
#include <vector>

namespace hnsw_concurrency_unittest {

using innodb_vector::HnswIndex;
using innodb_vector::hnsw_config_t;
using innodb_vector::hnsw_metric_t;

static std::vector<float> random_vector(size_t dims, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(dims);
  for (size_t i = 0; i < dims; ++i) v[i] = dist(rng);
  return v;
}

class HnswConcurrencyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    config_.M = 16;
    config_.M0 = 32;
    config_.ef_construction = 50;
    config_.ef_search = 30;
    config_.max_elements = 5000;
    config_.dimensions = 32;  // Small dims for speed
    config_.metric = hnsw_metric_t::L2;
  }

  void build_index(HnswIndex &index, uint64_t start_id, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
      auto v = random_vector(config_.dimensions,
                              static_cast<unsigned>(start_id + i));
      index.insert(start_id + i, v);
    }
  }

  hnsw_config_t config_;
};

// ----------------------------------------------------------------------------
// Test 1: Concurrent reads (shared_lock allows parallel searches)
// ----------------------------------------------------------------------------
TEST_F(HnswConcurrencyTest, ConcurrentSearches) {
  HnswIndex index(config_);
  build_index(index, 1, 500);

  const int num_threads = 4;
  const int searches_per_thread = 100;
  std::atomic<int> valid_results{0};

  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t]() {
      for (int i = 0; i < searches_per_thread; ++i) {
        auto q = random_vector(config_.dimensions,
                                static_cast<unsigned>(10000 + t * 1000 + i));
        auto results = index.search(q, 5);
        if (!results.empty()) ++valid_results;
      }
    });
  }

  for (auto &th : threads) th.join();

  EXPECT_EQ(valid_results.load(), num_threads * searches_per_thread);
}

// ----------------------------------------------------------------------------
// Test 2: Concurrent inserts (exclusive lock serializes writes)
// ----------------------------------------------------------------------------
TEST_F(HnswConcurrencyTest, ConcurrentInserts) {
  HnswIndex index(config_);

  const int num_threads = 4;
  const int inserts_per_thread = 100;
  std::atomic<int> successes{0};

  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t]() {
      for (int i = 0; i < inserts_per_thread; ++i) {
        uint64_t id = static_cast<uint64_t>(t * 10000 + i);
        auto v = random_vector(config_.dimensions, static_cast<unsigned>(id));
        if (index.insert(id, v)) ++successes;
      }
    });
  }

  for (auto &th : threads) th.join();

  EXPECT_EQ(successes.load(), num_threads * inserts_per_thread);
  EXPECT_EQ(index.size(),
            static_cast<uint64_t>(num_threads * inserts_per_thread));

  // Verify all IDs present
  for (int t = 0; t < num_threads; ++t) {
    for (int i = 0; i < inserts_per_thread; ++i) {
      EXPECT_TRUE(index.contains(static_cast<uint64_t>(t * 10000 + i)));
    }
  }
}

// ----------------------------------------------------------------------------
// Test 3: Insert during search (read-write contention)
// ----------------------------------------------------------------------------
TEST_F(HnswConcurrencyTest, InsertDuringSearch) {
  HnswIndex index(config_);
  build_index(index, 1, 200);

  std::atomic<int> search_count{0};
  std::atomic<int> insert_count{0};
  std::atomic<bool> stop_readers{false};

  std::vector<std::thread> threads;

  // 2 reader threads
  for (int t = 0; t < 2; ++t) {
    threads.emplace_back([&, t]() {
      while (!stop_readers.load()) {
        auto q = random_vector(config_.dimensions,
                                static_cast<unsigned>(50000 + search_count));
        auto results = index.search(q, 5);
        (void)results;
        ++search_count;
      }
    });
  }

  // 2 writer threads
  for (int t = 0; t < 2; ++t) {
    threads.emplace_back([&, t]() {
      for (int i = 0; i < 100; ++i) {
        uint64_t id = static_cast<uint64_t>(1000 + t * 1000 + i);
        auto v = random_vector(config_.dimensions, static_cast<unsigned>(id));
        if (index.insert(id, v)) ++insert_count;
      }
    });
  }

  // Wait for writers to finish
  threads[2].join();
  threads[3].join();

  // Stop readers
  stop_readers.store(true);
  threads[0].join();
  threads[1].join();

  EXPECT_EQ(insert_count.load(), 200);
  EXPECT_GT(search_count.load(), 0);
  EXPECT_EQ(index.size(), 400u);  // 200 initial + 200 inserted
}

// ----------------------------------------------------------------------------
// Test 4: Remove during search
// ----------------------------------------------------------------------------
TEST_F(HnswConcurrencyTest, RemoveDuringSearch) {
  HnswIndex index(config_);
  build_index(index, 1, 500);

  std::atomic<int> search_count{0};
  std::atomic<int> remove_count{0};
  std::atomic<bool> stop_readers{false};

  std::vector<std::thread> threads;

  // 2 reader threads
  for (int t = 0; t < 2; ++t) {
    threads.emplace_back([&, t]() {
      while (!stop_readers.load()) {
        auto q = random_vector(config_.dimensions,
                                static_cast<unsigned>(60000 + search_count));
        auto results = index.search(q, 5);
        (void)results;
        ++search_count;
      }
    });
  }

  // 2 remover threads (remove IDs 1-250 and 251-500)
  for (int t = 0; t < 2; ++t) {
    threads.emplace_back([&, t]() {
      uint64_t start = static_cast<uint64_t>(t * 250 + 1);
      for (uint64_t i = start; i < start + 250; ++i) {
        if (index.remove(i)) ++remove_count;
      }
    });
  }

  threads[2].join();
  threads[3].join();
  stop_readers.store(true);
  threads[0].join();
  threads[1].join();

  EXPECT_EQ(remove_count.load(), 500);
  EXPECT_EQ(index.size(), 0u);
  EXPECT_GT(search_count.load(), 0);
}

// ----------------------------------------------------------------------------
// Test 5: Update during search
// ----------------------------------------------------------------------------
TEST_F(HnswConcurrencyTest, UpdateDuringSearch) {
  HnswIndex index(config_);
  build_index(index, 1, 200);

  std::atomic<int> search_count{0};
  std::atomic<int> update_count{0};
  std::atomic<bool> stop_readers{false};

  std::vector<std::thread> threads;

  // 2 readers
  for (int t = 0; t < 2; ++t) {
    threads.emplace_back([&, t]() {
      while (!stop_readers.load()) {
        auto q = random_vector(config_.dimensions,
                                static_cast<unsigned>(70000 + search_count));
        auto results = index.search(q, 5);
        (void)results;
        ++search_count;
      }
    });
  }

  // 2 updaters (update existing IDs with new vectors)
  for (int t = 0; t < 2; ++t) {
    threads.emplace_back([&, t]() {
      uint64_t start = static_cast<uint64_t>(t * 100 + 1);
      for (uint64_t i = start; i < start + 100; ++i) {
        auto v = random_vector(config_.dimensions,
                                static_cast<unsigned>(80000 + i));
        if (index.update(i, v)) ++update_count;
      }
    });
  }

  threads[2].join();
  threads[3].join();
  stop_readers.store(true);
  threads[0].join();
  threads[1].join();

  EXPECT_EQ(update_count.load(), 200);
  EXPECT_EQ(index.size(), 200u);
  EXPECT_GT(search_count.load(), 0);
}

// ----------------------------------------------------------------------------
// Test 6: Concurrent insert and remove
// ----------------------------------------------------------------------------
TEST_F(HnswConcurrencyTest, ConcurrentInsertAndRemove) {
  HnswIndex index(config_);
  // Pre-fill with IDs 1-200
  build_index(index, 1, 200);

  std::atomic<int> inserted{0};
  std::atomic<int> removed{0};

  std::vector<std::thread> threads;

  // 2 inserters: add IDs 201-400
  for (int t = 0; t < 2; ++t) {
    threads.emplace_back([&, t]() {
      uint64_t start = static_cast<uint64_t>(201 + t * 100);
      for (uint64_t i = start; i < start + 100; ++i) {
        auto v = random_vector(config_.dimensions, static_cast<unsigned>(i));
        if (index.insert(i, v)) ++inserted;
      }
    });
  }

  // 2 removers: remove IDs 1-200
  for (int t = 0; t < 2; ++t) {
    threads.emplace_back([&, t]() {
      uint64_t start = static_cast<uint64_t>(t * 100 + 1);
      for (uint64_t i = start; i < start + 100; ++i) {
        if (index.remove(i)) ++removed;
      }
    });
  }

  for (auto &th : threads) th.join();

  EXPECT_EQ(inserted.load(), 200);
  EXPECT_EQ(removed.load(), 200);
  // Final size: 200 (initial) + 200 (inserted) - 200 (removed) = 200
  EXPECT_EQ(index.size(), 200u);

  // Verify new IDs present, old IDs gone
  for (uint64_t i = 1; i <= 200; ++i) {
    EXPECT_FALSE(index.contains(i));
  }
  for (uint64_t i = 201; i <= 400; ++i) {
    EXPECT_TRUE(index.contains(i));
  }
}

// ----------------------------------------------------------------------------
// Test 7: Stress test (mixed operations, many threads)
// ----------------------------------------------------------------------------
TEST_F(HnswConcurrencyTest, StressTest) {
  HnswIndex index(config_);
  build_index(index, 1, 100);

  const int num_threads = 8;
  const int ops_per_thread = 200;
  std::atomic<int> errors{0};

  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t]() {
      std::mt19937 rng(static_cast<unsigned>(t * 7919));
      std::uniform_int_distribution<int> op_dist(0, 3);

      for (int i = 0; i < ops_per_thread; ++i) {
        uint64_t id = static_cast<uint64_t>(t * 10000 + i);
        auto v = random_vector(config_.dimensions,
                                static_cast<unsigned>(id + 99999));

        switch (op_dist(rng)) {
          case 0:  // insert
            index.insert(id, v);
            break;
          case 1:  // search
          {
            auto results = index.search(v, 3);
            (void)results;
            break;
          }
          case 2:  // remove
            index.remove(id);
            break;
          case 3:  // contains
            index.contains(id);
            break;
        }
      }
    });
  }

  for (auto &th : threads) th.join();

  // No crash is the primary assertion.
  // Note: size()/deleted_count()/total_nodes() are unlocked getters,
  // so exact consistency after concurrent ops is not guaranteed.
  // A search on the final state should still work without crashing.
  auto q = random_vector(config_.dimensions, 777);
  auto results = index.search(q, 5);
  (void)results;
  SUCCEED() << "Stress test completed without crashes";
}

// ----------------------------------------------------------------------------
// Test 8: Concurrent contains during inserts
// ----------------------------------------------------------------------------
TEST_F(HnswConcurrencyTest, ConcurrentContains) {
  HnswIndex index(config_);

  std::atomic<int> found{0};
  std::atomic<int> not_found{0};
  std::atomic<bool> stop_readers{false};

  std::vector<std::thread> threads;

  // 2 inserters
  for (int t = 0; t < 2; ++t) {
    threads.emplace_back([&, t]() {
      for (int i = 0; i < 200; ++i) {
        uint64_t id = static_cast<uint64_t>(t * 10000 + i);
        auto v = random_vector(config_.dimensions, static_cast<unsigned>(id));
        index.insert(id, v);
      }
    });
  }

  // 4 readers checking contains()
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&, t]() {
      std::mt19937 rng(static_cast<unsigned>(t + 42));
      std::uniform_int_distribution<uint64_t> id_dist(0, 20000);
      while (!stop_readers.load()) {
        uint64_t id = id_dist(rng);
        if (index.contains(id))
          ++found;
        else
          ++not_found;
      }
    });
  }

  // Wait for inserters
  threads[0].join();
  threads[1].join();

  // Stop readers
  stop_readers.store(true);
  for (int t = 2; t < 6; ++t) threads[t].join();

  EXPECT_GT(found.load() + not_found.load(), 0);
  EXPECT_EQ(index.size(), 400u);
}

}  // namespace hnsw_concurrency_unittest
