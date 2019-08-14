#include <memory>
#include <random>
// uncomment to disable assert()
// #define NDEBUG
#include <cassert>
#include <sstream>

#include <clipper/metrics.hpp>
#include <clipper/task_executor.hpp>
#include <clipper/util.hpp>
#include <clipper/logging.hpp>

namespace clipper {

CacheEntry::CacheEntry() {}

PredictionCache::PredictionCache(size_t size_bytes)
    : max_size_bytes_(size_bytes) {
  lookups_counter_ = metrics::MetricsRegistry::get_metrics().create_counter(
      "internal:prediction_cache_lookups");
  hit_ratio_ = metrics::MetricsRegistry::get_metrics().create_ratio_counter(
      "internal:prediction_cache_hit_ratio");
}

std::pair<folly::Future<Output>, folly::Future<Output>> PredictionCache::fetch(PredictTask& t) {
  CacheEntry new_entry;
  // create promise/future pair for this request
  folly::Promise<Output> new_promise;
  folly::Future<Output> new_future = new_promise.getFuture();

  folly::Promise<Output> new_backup_promise;
  folly::Future<Output> new_backup_future = new_backup_promise.getFuture();

  new_entry.value_promises_.push_back(std::move(new_promise));
  new_entry.backup_value_promises_.push_back(std::move(new_backup_promise));

  std::unique_lock<std::mutex> l(m_);
  insert_entry(t, new_entry);
  hit_ratio_->increment(0, 1);

  return std::make_pair(std::move(new_future), std::move(new_backup_future));
}

void PredictionCache::put(size_t cache_id,
                          const Output &output,
                          bool redundant) {
  std::unique_lock<std::mutex> l(m_);
  auto search = entries_.find(cache_id);
  if (search != entries_.end()) {
    CacheEntry &entry = search->second;
    if (!entry.completed_) {
      if (redundant) {
        for (auto &p : entry.backup_value_promises_) {
          p.setValue(std::move(output));
        }
        entry.backup_completed_ = true;
        entry.backup_value_ = output;
      } else {
        // Complete the outstanding promises
        for (auto &p : entry.value_promises_) {
          p.setValue(std::move(output));
        }
        entry.completed_ = true;
        entry.value_ = output;
      }
    }
  } else {
    log_error_formatted(LOGGING_TAG_PARM, "Trying to put() evicted entry {}", cache_id);
  }
}

// Mutex must be held when calling
void PredictionCache::insert_entry(PredictTask& t, CacheEntry &value) {
  if (to_evict_list_.size() == max_entries_) {
    auto to_evict_id = to_evict_list_.front();
    to_evict_list_.pop();
    entries_.erase(to_evict_id);
  }

  t.cache_id_ = cur_id_++;
  to_evict_list_.push(t.cache_id_);
  entries_.insert(std::make_pair(t.cache_id_, std::move(value)));
}

void PredictionCache::evict_entries(long space_needed_bytes) {
  if (space_needed_bytes <= 0) {
    return;
  }
  while (space_needed_bytes > 0 && !page_buffer_.empty()) {
    long page_key = page_buffer_[page_buffer_index_];
    auto page_entry_search = entries_.find(page_key);
    if (page_entry_search == entries_.end()) {
      throw std::runtime_error(
          "Failed to find corresponding cache entry for a buffer page!");
    }
    CacheEntry &page_entry = page_entry_search->second;
    if (page_entry.used_ || !page_entry.completed_) {
      page_entry.used_ = false;
      page_buffer_index_ = (page_buffer_index_ + 1) % page_buffer_.size();
    } else {
      page_buffer_.erase(page_buffer_.begin() + page_buffer_index_);
      page_buffer_index_ = page_buffer_.size() > 0
                               ? page_buffer_index_ % page_buffer_.size()
                               : 0;
      size_bytes_ -= page_entry.value_.y_hat_->byte_size();
      space_needed_bytes -= page_entry.value_.y_hat_->byte_size();
      entries_.erase(page_entry_search);
    }
  }
}

size_t PredictionCache::hash(const VersionedModelId &model,
                             size_t input_hash) const {
  std::size_t seed = 0;
  size_t model_hash = std::hash<clipper::VersionedModelId>()(model);
  boost::hash_combine(seed, model_hash);
  boost::hash_combine(seed, input_hash);
  return seed;
}

}  // namespace clipper
