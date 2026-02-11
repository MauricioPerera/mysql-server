/**
  @file storage/innobase/vector/vec0hnsw_registry.cc

  HNSW Index Registry Implementation.
  Supports multiple indexes per table via table:column composite keys.
  Supports auto-persistence via IndexEntry with file_path.
*/

#include "../include/vec0hnsw_registry.h"
#include <algorithm>
#include <cctype>
#include <cstdio>

namespace innodb_vector {

HnswIndexRegistry& HnswIndexRegistry::instance() {
  static HnswIndexRegistry registry;
  return registry;
}

bool HnswIndexRegistry::register_index(const std::string& table_name,
                                        const std::string& column_name,
                                        size_t dim, size_t M,
                                        size_t ef_construction,
                                        hnsw_metric_t metric) {
  std::lock_guard<std::mutex> lock(mutex_);

  std::string key = make_key(table_name, column_name);
  if (indexes_.find(key) != indexes_.end()) {
    return false;  // Index already exists
  }

  hnsw_config_t config;
  config.dimensions = static_cast<uint32_t>(dim);
  config.M = static_cast<uint32_t>(M);
  config.M0 = static_cast<uint32_t>(M * 2);
  config.ef_construction = static_cast<uint32_t>(ef_construction);
  config.metric = metric;

  IndexEntry entry;
  entry.index = std::make_shared<HnswIndex>(config);
  indexes_[key] = std::move(entry);
  return true;
}

bool HnswIndexRegistry::register_loaded_index(
    const std::string& table_name,
    const std::string& column_name,
    std::shared_ptr<HnswIndex> index,
    const std::string& file_path) {
  std::lock_guard<std::mutex> lock(mutex_);

  std::string key = make_key(table_name, column_name);
  if (indexes_.find(key) != indexes_.end()) {
    return false;  // Already exists
  }

  IndexEntry entry;
  entry.index = std::move(index);
  entry.file_path = file_path;
  indexes_[key] = std::move(entry);
  return true;
}

std::shared_ptr<HnswIndex> HnswIndexRegistry::get_index(
    const std::string& table_name,
    const std::string& column_name) {
  std::lock_guard<std::mutex> lock(mutex_);

  std::string key = make_key(table_name, column_name);
  auto it = indexes_.find(key);
  if (it != indexes_.end()) {
    return it->second.index;
  }

  /* When column is empty (legacy lookup), fall back to finding any
     index registered with a "table:column" key.  This handles the
     common case where HNSW_INFO('table') is called for a DD-declared
     index that was registered with the column name. */
  if (column_name.empty()) {
    std::string prefix = table_name + ":";
    for (const auto &pair : indexes_) {
      if (pair.first.compare(0, prefix.size(), prefix) == 0) {
        return pair.second.index;
      }
    }
  }

  return nullptr;
}

bool HnswIndexRegistry::drop_index(const std::string& table_name,
                                    const std::string& column_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::string key = make_key(table_name, column_name);
  auto it = indexes_.find(key);

  if (it == indexes_.end() && column_name.empty()) {
    std::string prefix = table_name + ":";
    for (it = indexes_.begin(); it != indexes_.end(); ++it) {
      if (it->first.compare(0, prefix.size(), prefix) == 0) break;
    }
  }

  if (it == indexes_.end()) return false;

  // Delete .hnsw file from disk if path is known
  if (!it->second.file_path.empty()) {
    std::remove(it->second.file_path.c_str());
  }

  indexes_.erase(it);
  return true;
}

bool HnswIndexRegistry::has_index(const std::string& table_name,
                                   const std::string& column_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::string key = make_key(table_name, column_name);
  if (indexes_.find(key) != indexes_.end()) return true;

  if (column_name.empty()) {
    std::string prefix = table_name + ":";
    for (const auto &pair : indexes_) {
      if (pair.first.compare(0, prefix.size(), prefix) == 0) return true;
    }
  }
  return false;
}

std::vector<std::string> HnswIndexRegistry::list_indexes() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::string> result;
  result.reserve(indexes_.size());
  for (const auto& pair : indexes_) {
    result.push_back(pair.first);
  }
  return result;
}

std::vector<std::string> HnswIndexRegistry::get_columns_for_table(
    const std::string& table_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::string> result;
  std::string prefix = table_name + ":";

  for (const auto& pair : indexes_) {
    if (pair.first == table_name) {
      // Legacy entry (no column)
      result.push_back("");
    } else if (pair.first.compare(0, prefix.size(), prefix) == 0) {
      // table:column entry
      result.push_back(pair.first.substr(prefix.size()));
    }
  }
  return result;
}

void HnswIndexRegistry::set_file_path(const std::string& table_name,
                                       const std::string& column_name,
                                       const std::string& path) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::string key = make_key(table_name, column_name);
  auto it = indexes_.find(key);

  if (it == indexes_.end() && column_name.empty()) {
    std::string prefix = table_name + ":";
    for (it = indexes_.begin(); it != indexes_.end(); ++it) {
      if (it->first.compare(0, prefix.size(), prefix) == 0) break;
    }
  }

  if (it != indexes_.end()) {
    it->second.file_path = path;
  }
}

std::string HnswIndexRegistry::get_file_path(const std::string& table_name,
                                              const std::string& column_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::string key = make_key(table_name, column_name);
  auto it = indexes_.find(key);
  if (it != indexes_.end()) {
    return it->second.file_path;
  }

  if (column_name.empty()) {
    std::string prefix = table_name + ":";
    for (const auto &pair : indexes_) {
      if (pair.first.compare(0, prefix.size(), prefix) == 0) {
        return pair.second.file_path;
      }
    }
  }
  return "";
}

void HnswIndexRegistry::replace_index(const std::string& table_name,
                                       const std::string& column_name,
                                       std::shared_ptr<HnswIndex> new_index) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::string key = make_key(table_name, column_name);
  auto it = indexes_.find(key);
  if (it != indexes_.end()) {
    it->second.index = std::move(new_index);
    it->second.needs_reconcile = false;
  }
}

void HnswIndexRegistry::set_needs_reconcile(const std::string& table_name,
                                             const std::string& column_name,
                                             bool flag) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::string key = make_key(table_name, column_name);
  auto it = indexes_.find(key);
  if (it != indexes_.end()) {
    it->second.needs_reconcile = flag;
  }
}

bool HnswIndexRegistry::check_and_clear_reconcile(
    const std::string& table_name, const std::string& column_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::string key = make_key(table_name, column_name);
  auto it = indexes_.find(key);
  if (it != indexes_.end() && it->second.needs_reconcile) {
    it->second.needs_reconcile = false;
    return true;
  }
  return false;
}

size_t HnswIndexRegistry::save_all_dirty() {
  /* Collect dirty indexes under the lock, then release it before
     doing file I/O so that concurrent DML isn't blocked. */
  std::vector<std::pair<std::shared_ptr<HnswIndex>, std::string>> to_save;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& pair : indexes_) {
      auto& entry = pair.second;
      if (entry.index && entry.index->is_dirty() && !entry.file_path.empty()) {
        to_save.emplace_back(entry.index, entry.file_path);
      }
    }
  }

  size_t saved = 0;
  for (auto& [index, path] : to_save) {
    if (index->save_to_file(path.c_str())) {
      index->mark_clean();
      ++saved;
    }
  }
  return saved;
}

hnsw_metric_t HnswIndexRegistry::parse_metric(const std::string& metric_str) {
  std::string lower;
  lower.reserve(metric_str.size());
  for (char c : metric_str) {
    lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }

  if (lower == "cosine" || lower == "cos") {
    return hnsw_metric_t::COSINE;
  } else if (lower == "dot_product" || lower == "dot" || lower == "ip" ||
             lower == "inner_product") {
    return hnsw_metric_t::DOT_PRODUCT;
  }
  // Default: L2
  return hnsw_metric_t::L2;
}

const char* HnswIndexRegistry::metric_to_string(hnsw_metric_t metric) {
  switch (metric) {
    case hnsw_metric_t::COSINE:
      return "cosine";
    case hnsw_metric_t::DOT_PRODUCT:
      return "dot_product";
    case hnsw_metric_t::L2:
    default:
      return "l2";
  }
}

void HnswIndexRegistry::parse_comment(const char *comment,
                                       uint32_t *M, uint32_t *ef_construction,
                                       hnsw_metric_t *metric) {
  *M = 16;
  *ef_construction = 200;
  *metric = hnsw_metric_t::L2;
  if (!comment || !*comment) return;

  std::string s(comment);
  unsigned val;

  const char *m_ptr = strstr(s.c_str(), "M=");
  if (m_ptr && sscanf(m_ptr, "M=%u", &val) == 1) {
    *M = val;
  }

  const char *ef_ptr = strstr(s.c_str(), "ef=");
  if (ef_ptr && sscanf(ef_ptr, "ef=%u", &val) == 1) {
    *ef_construction = val;
  }

  const char *metric_ptr = strstr(s.c_str(), "metric=");
  if (metric_ptr) {
    std::string metric_str(metric_ptr + 7);
    auto comma = metric_str.find(',');
    if (comma != std::string::npos) metric_str.resize(comma);
    *metric = parse_metric(metric_str);
  }
}

}  // namespace innodb_vector
