/**
  @file sql/item_hnsw_func.cc

  HNSW Index Management SQL Functions Implementation.
  Supports optional column parameter for multi-index per table.
  Supports optional metric parameter for distance metric selection.

  HNSW_CREATE_INDEX signatures:
    4 args: (table, dim, M, ef)                    -- legacy, L2
    5 args: (table, column, dim, M, ef)            -- multi-index, L2
         OR (table, dim, M, ef, metric)            -- legacy, custom metric
    6 args: (table, column, dim, M, ef, metric)    -- multi-index, custom metric

  Disambiguation for 5 args:
    - If arg[1] is STRING and arg[4] is INT  -> (table, column, dim, M, ef)
    - If arg[1] is INT and arg[4] is STRING  -> (table, dim, M, ef, metric)
*/

#include "sql/item_strfunc.h"
#include "sql/mysqld.h"
#include "mysqld_error.h"
#include "storage/innobase/include/vec0hnsw_registry.h"

#include <cstdio>
#include <sstream>

/** Strip schema prefix from table name.
    "db.table" → "table", "db/table" → "table", "table" → "table".
    The registry stores bare table names (from table->s->table_name.str). */
static std::string normalize_table_name(const std::string &name) {
  std::string result = name;
  auto dot = result.rfind('.');
  if (dot != std::string::npos) result = result.substr(dot + 1);
  auto slash = result.rfind('/');
  if (slash != std::string::npos) result = result.substr(slash + 1);
  return result;
}

// ============================================================================
// HNSW_CREATE_INDEX Implementation
// ============================================================================

bool Item_func_hnsw_create_index::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 1, MYSQL_TYPE_VARCHAR)) return true;

  if (arg_count == 6) {
    // (table, column, dim, M, ef, metric)
    if (param_type_is_default(thd, 1, 2, MYSQL_TYPE_VARCHAR)) return true;
    if (param_type_is_default(thd, 2, 3, MYSQL_TYPE_LONG)) return true;
    if (param_type_is_default(thd, 3, 4, MYSQL_TYPE_LONG)) return true;
    if (param_type_is_default(thd, 4, 5, MYSQL_TYPE_LONG)) return true;
    if (param_type_is_default(thd, 5, 6, MYSQL_TYPE_VARCHAR)) return true;
  } else if (arg_count == 5) {
    // Ambiguous: could be (table, column, dim, M, ef) or (table, dim, M, ef, metric)
    // Don't force types on args 1 and 4 - disambiguate at runtime
    if (param_type_is_default(thd, 2, 3, MYSQL_TYPE_LONG)) return true;
    if (param_type_is_default(thd, 3, 4, MYSQL_TYPE_LONG)) return true;
  } else {
    // 4 args: (table, dim, M, ef)
    if (param_type_is_default(thd, 1, 2, MYSQL_TYPE_LONG)) return true;
    if (param_type_is_default(thd, 2, 3, MYSQL_TYPE_LONG)) return true;
    if (param_type_is_default(thd, 3, 4, MYSQL_TYPE_LONG)) return true;
  }
  set_data_type_string(255U);
  set_nullable(true);
  return false;
}

String *Item_func_hnsw_create_index::val_str(String *) {
  assert(fixed);

  String table_buf;
  String *table_str = args[0]->val_str(&table_buf);
  if (!table_str) { null_value = true; return nullptr; }

  std::string table_name = normalize_table_name(table_str->c_ptr_safe());
  std::string column_name;
  longlong dim, M, ef;
  innodb_vector::hnsw_metric_t metric = innodb_vector::hnsw_metric_t::L2;

  if (arg_count == 6) {
    // (table, column, dim, M, ef, metric)
    String col_buf;
    String *col_str = args[1]->val_str(&col_buf);
    if (!col_str) { null_value = true; return nullptr; }
    column_name = std::string(col_str->c_ptr_safe());
    dim = args[2]->val_int();
    M = args[3]->val_int();
    ef = args[4]->val_int();
    String metric_buf;
    String *metric_str = args[5]->val_str(&metric_buf);
    if (metric_str) {
      metric = innodb_vector::HnswIndexRegistry::parse_metric(
          std::string(metric_str->c_ptr_safe()));
    }
  } else if (arg_count == 5) {
    // Disambiguate: arg[1] STRING -> column mode; arg[4] STRING -> metric mode
    if (args[1]->result_type() == STRING_RESULT &&
        args[4]->result_type() == INT_RESULT) {
      // (table, column, dim, M, ef)
      String col_buf;
      String *col_str = args[1]->val_str(&col_buf);
      if (!col_str) { null_value = true; return nullptr; }
      column_name = std::string(col_str->c_ptr_safe());
      dim = args[2]->val_int();
      M = args[3]->val_int();
      ef = args[4]->val_int();
    } else {
      // (table, dim, M, ef, metric)
      dim = args[1]->val_int();
      M = args[2]->val_int();
      ef = args[3]->val_int();
      String metric_buf;
      String *metric_str = args[4]->val_str(&metric_buf);
      if (metric_str) {
        metric = innodb_vector::HnswIndexRegistry::parse_metric(
            std::string(metric_str->c_ptr_safe()));
      }
    }
  } else {
    // 4 args: (table, dim, M, ef)
    dim = args[1]->val_int();
    M = args[2]->val_int();
    ef = args[3]->val_int();
  }

  if (dim <= 0 || dim > 16383) {
    result_buffer.set_ascii("ERROR: dim must be between 1 and 16383", 38);
    null_value = false;
    return &result_buffer;
  }
  if (M <= 0 || M > 128) {
    result_buffer.set_ascii("ERROR: M must be between 1 and 128", 35);
    null_value = false;
    return &result_buffer;
  }

  auto& registry = innodb_vector::HnswIndexRegistry::instance();
  bool success = registry.register_index(table_name, column_name,
                                          static_cast<size_t>(dim),
                                          static_cast<size_t>(M),
                                          static_cast<size_t>(ef),
                                          metric);

  if (success) {
    std::string msg = "OK: Index created (metric=" +
        std::string(innodb_vector::HnswIndexRegistry::metric_to_string(metric)) + ")";
    result_buffer.copy(msg.c_str(), msg.length(), &my_charset_utf8mb4_bin);
  } else {
    result_buffer.set_ascii("ERROR: Index already exists", 27);
  }

  null_value = false;
  return &result_buffer;
}

// ============================================================================
// HNSW_DROP_INDEX Implementation
// 1 arg: (table)           -- legacy
// 2 args: (table, column)  -- multi-index
// ============================================================================

bool Item_func_hnsw_drop_index::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 1, MYSQL_TYPE_VARCHAR)) return true;
  if (arg_count >= 2) {
    if (param_type_is_default(thd, 1, 2, MYSQL_TYPE_VARCHAR)) return true;
  }
  set_data_type_string(255U);
  set_nullable(true);
  return false;
}

String *Item_func_hnsw_drop_index::val_str(String *) {
  assert(fixed);

  String table_buf;
  String *table_str = args[0]->val_str(&table_buf);
  if (!table_str) { null_value = true; return nullptr; }

  std::string table_name = normalize_table_name(table_str->c_ptr_safe());
  std::string column_name;

  if (arg_count >= 2) {
    String col_buf;
    String *col_str = args[1]->val_str(&col_buf);
    if (col_str) column_name = std::string(col_str->c_ptr_safe());
  }

  auto& registry = innodb_vector::HnswIndexRegistry::instance();
  bool success = registry.drop_index(table_name, column_name);

  if (success) {
    result_buffer.set_ascii("OK: Index dropped", 17);
  } else {
    result_buffer.set_ascii("ERROR: Index not found", 22);
  }

  null_value = false;
  return &result_buffer;
}

// ============================================================================
// HNSW_SAVE_INDEX Implementation
// 2 args: (table, path)           -- legacy
// 3 args: (table, column, path)   -- multi-index
// ============================================================================

bool Item_func_hnsw_save_index::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 1, MYSQL_TYPE_VARCHAR)) return true;
  if (param_type_is_default(thd, 1, 2, MYSQL_TYPE_VARCHAR)) return true;
  if (arg_count >= 3) {
    if (param_type_is_default(thd, 2, 3, MYSQL_TYPE_VARCHAR)) return true;
  }
  set_data_type_string(255U);
  set_nullable(true);
  return false;
}

String *Item_func_hnsw_save_index::val_str(String *) {
  assert(fixed);

  String table_buf;
  String *table_str = args[0]->val_str(&table_buf);
  if (!table_str) { null_value = true; return nullptr; }

  std::string table_name = normalize_table_name(table_str->c_ptr_safe());
  std::string column_name;
  std::string path;

  if (arg_count == 3) {
    String col_buf, path_buf;
    String *col_str = args[1]->val_str(&col_buf);
    String *path_str = args[2]->val_str(&path_buf);
    if (!col_str || !path_str) { null_value = true; return nullptr; }
    column_name = std::string(col_str->c_ptr_safe());
    path = std::string(path_str->c_ptr_safe());
  } else {
    String path_buf;
    String *path_str = args[1]->val_str(&path_buf);
    if (!path_str) { null_value = true; return nullptr; }
    path = std::string(path_str->c_ptr_safe());
  }

  auto& registry = innodb_vector::HnswIndexRegistry::instance();
  auto index = registry.get_index(table_name, column_name);

  if (!index) {
    result_buffer.set_ascii("ERROR: Index not found", 22);
  } else if (index->save_to_file(path.c_str())) {
    std::ostringstream msg;
    msg << "OK: Index saved (" << index->size() << " vectors)";
    std::string msg_str = msg.str();
    result_buffer.copy(msg_str.c_str(), msg_str.length(), &my_charset_utf8mb4_bin);
  } else {
    result_buffer.set_ascii("ERROR: Save failed", 18);
  }

  null_value = false;
  return &result_buffer;
}

// ============================================================================
// HNSW_LOAD_INDEX Implementation
// 2 args: (table, path)           -- legacy
// 3 args: (table, column, path)   -- multi-index
// ============================================================================

bool Item_func_hnsw_load_index::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 1, MYSQL_TYPE_VARCHAR)) return true;
  if (param_type_is_default(thd, 1, 2, MYSQL_TYPE_VARCHAR)) return true;
  if (arg_count >= 3) {
    if (param_type_is_default(thd, 2, 3, MYSQL_TYPE_VARCHAR)) return true;
  }
  set_data_type_string(255U);
  set_nullable(true);
  return false;
}

String *Item_func_hnsw_load_index::val_str(String *) {
  assert(fixed);

  String table_buf;
  String *table_str = args[0]->val_str(&table_buf);
  if (!table_str) { null_value = true; return nullptr; }

  std::string table_name = normalize_table_name(table_str->c_ptr_safe());
  std::string column_name;
  std::string path;

  if (arg_count == 3) {
    String col_buf, path_buf;
    String *col_str = args[1]->val_str(&col_buf);
    String *path_str = args[2]->val_str(&path_buf);
    if (!col_str || !path_str) { null_value = true; return nullptr; }
    column_name = std::string(col_str->c_ptr_safe());
    path = std::string(path_str->c_ptr_safe());
  } else {
    String path_buf;
    String *path_str = args[1]->val_str(&path_buf);
    if (!path_str) { null_value = true; return nullptr; }
    path = std::string(path_str->c_ptr_safe());
  }

  auto& registry = innodb_vector::HnswIndexRegistry::instance();
  auto index = registry.get_index(table_name, column_name);

  if (!index) {
    result_buffer.set_ascii("ERROR: Index not found (create first)", 37);
  } else if (index->load_from_file(path.c_str())) {
    std::ostringstream msg;
    msg << "OK: Index loaded (" << index->size() << " vectors)";
    std::string msg_str = msg.str();
    result_buffer.copy(msg_str.c_str(), msg_str.length(), &my_charset_utf8mb4_bin);
  } else {
    result_buffer.set_ascii("ERROR: Load failed", 18);
  }

  null_value = false;
  return &result_buffer;
}

// ============================================================================
// HNSW_INFO Implementation
// 1 arg:  (table)           -- legacy single-index
// 2 args: (table, column)   -- multi-index
// Returns JSON with index metadata.
// ============================================================================

bool Item_func_hnsw_info::resolve_type(THD *thd) {
  if (param_type_is_default(thd, 0, 1, MYSQL_TYPE_VARCHAR)) return true;
  if (arg_count >= 2) {
    if (param_type_is_default(thd, 1, 2, MYSQL_TYPE_VARCHAR)) return true;
  }
  set_data_type_string(4096U);
  set_nullable(true);
  return false;
}

String *Item_func_hnsw_info::val_str(String *) {
  assert(fixed);

  String table_buf;
  String *table_str = args[0]->val_str(&table_buf);
  if (!table_str) { null_value = true; return nullptr; }

  std::string table_name = normalize_table_name(table_str->c_ptr_safe());
  std::string column_name;

  if (arg_count >= 2) {
    String col_buf;
    String *col_str = args[1]->val_str(&col_buf);
    if (col_str) column_name = std::string(col_str->c_ptr_safe());
  }

  auto& registry = innodb_vector::HnswIndexRegistry::instance();
  auto index = registry.get_index(table_name, column_name);

  if (!index) {
    null_value = true;
    return nullptr;
  }

  const auto& cfg = index->config();
  std::string file_path = registry.get_file_path(table_name, column_name);

  // Build JSON manually (no external dependency)
  std::ostringstream json;
  json << "{";
  json << "\"table\": \"" << table_name << "\"";
  if (!column_name.empty()) {
    json << ", \"column\": \"" << column_name << "\"";
  }
  json << ", \"vectors\": " << index->size();
  json << ", \"deleted\": " << index->deleted_count();
  json << ", \"total_nodes\": " << index->total_nodes();
  json << ", \"dimensions\": " << cfg.dimensions;
  json << ", \"M\": " << cfg.M;
  json << ", \"ef_construction\": " << cfg.ef_construction;
  json << ", \"ef_search\": " << cfg.ef_search;
  json << ", \"metric\": \""
       << innodb_vector::HnswIndexRegistry::metric_to_string(cfg.metric)
       << "\"";
  json << ", \"max_elements\": " << cfg.max_elements;
  json << ", \"dirty\": " << (index->is_dirty() ? "true" : "false");
  if (!file_path.empty()) {
    json << ", \"file_path\": \"" << file_path << "\"";
  }
  json << "}";

  std::string result = json.str();
  result_buffer.copy(result.c_str(), result.length(), &my_charset_utf8mb4_bin);
  null_value = false;
  return &result_buffer;
}
