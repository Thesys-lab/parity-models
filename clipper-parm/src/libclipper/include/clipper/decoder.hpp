#ifndef CLIPPER_LIB_DECODER_H
#define CLIPPER_LIB_DECODER_H

#include <chrono>

#include <clipper/datatypes.hpp>
#include <clipper/logging.hpp>
#include <clipper/model_queue.hpp>

namespace clipper {

class DecodingGroup {
 public:

  DecodingGroup(unsigned int first_id, unsigned int group_size) :
    group_id_(first_id / group_size),
    first_id_(first_id),
    parity_id_(first_id + group_size - 1),
    group_size_(group_size) {
    cur_missing_ = first_id;
    for (unsigned int i = first_id + 1; i <= parity_id_; i++) {
      cur_missing_ ^= i;
    }
    decoding_buffer_.reserve(group_size);
    log_debug_formatted(LOGGING_TAG_PARM, "DecodingGroup::DecodingGroup({}, {})", first_id, group_size);
    start_time_ = std::chrono::system_clock::now();
  }

  void add_prediction(unsigned int id,
                      std::vector<std::shared_ptr<PredictionData>>&& predictions) {
    auto cur_time = std::chrono::system_clock::now();
    if (id == parity_id_) {
      // Record position of parity so that we can later use it in decoding.
      parity_pos_ = decoding_buffer_.size();
      parity_time_ = cur_time;
    }
    log_debug_formatted(LOGGING_TAG_PARM, "DecodingGroup({})::add_prediction({}) before cur_missing={}, have={}",
                       group_id_, id, cur_missing_, decoding_buffer_.size());
    cur_missing_ ^= id;
    decoding_buffer_.push_back(std::move(predictions));
    if (decoding_buffer_.size() == group_size_) {
      done_time_ = cur_time;
      long latency_micros =
        std::chrono::duration_cast<std::chrono::microseconds>(done_time_ - start_time_)
        .count();
      log_info_formatted(LOGGING_TAG_EC_METRICS, "DECODING_GROUP_DONE_LATENCY:group_id={},latency_micros={}",
                         group_id_, latency_micros);
    } else if (decoding_buffer_.size() == group_size_ - 1) {
      long latency_micros =
        std::chrono::duration_cast<std::chrono::microseconds>(cur_time - start_time_)
        .count();
      log_info_formatted(LOGGING_TAG_EC_METRICS, "DECODING_GROUP_READY_LATENCY:group_id={},latency_micros={}",
                         group_id_, latency_micros);

      bool parity_useful = (cur_missing_ != parity_id_);
      if (parity_useful) {
        latency_micros = std::chrono::duration_cast<std::chrono::microseconds>(cur_time - parity_time_).count();
      } else {
        latency_micros = std::chrono::duration_cast<std::chrono::microseconds>(parity_time_ - cur_time).count();
      }
      log_info_formatted(LOGGING_TAG_EC_METRICS, "DECODING_GROUP_PARITY_WAIT:group_id={},parity_useful={},latency_micros={}",
                         group_id_, parity_useful, latency_micros);
    }
    log_debug_formatted(LOGGING_TAG_PARM, "DecodingGroup({})::add_prediction({}) after cur_missing={}, have={}",
                       group_id_, id, cur_missing_, decoding_buffer_.size());
  }

  // Whether the DecodingGroup can be used for decoding a data unit.
  bool can_decode_data() {
    log_debug_formatted(LOGGING_TAG_PARM, "DecodingGroup({})::can_decode_data num_have={}, cur_missing={}, parity_id={}",
                       group_id_, decoding_buffer_.size(), cur_missing_, parity_id_);
    return decoding_buffer_.size() == (group_size_ - 1) && cur_missing_ != parity_id_;
  }

  bool done() { return decoding_buffer_.size() == group_size_; }

  bool parity_useless() {
    return decoding_buffer_.size() == (group_size_ - 1) && (cur_missing_ == parity_id_);
  }

  unsigned int group_id_;
  unsigned int first_id_;
  unsigned int parity_id_;
  unsigned int cur_missing_;
  unsigned int group_size_;
  std::vector<std::vector<std::shared_ptr<PredictionData>>> decoding_buffer_;
  std::chrono::time_point<std::chrono::system_clock> start_time_;
  std::chrono::time_point<std::chrono::system_clock> done_time_;
  std::chrono::time_point<std::chrono::system_clock> parity_time_;

  // Position of parity in decoding_buffer
  unsigned int parity_pos_;
};

class Decoder {
 public:
  Decoder(size_t num_models, size_t num_red_models, size_t ec_k_val,
          std::unordered_map<VersionedModelId, std::shared_ptr<ModelQueue>>* model_queues,
          QueueType queue_mode):
    num_models_(num_models),
    num_red_models_(num_red_models),
    ec_k_val_(ec_k_val),
    model_queues_(model_queues),
    queue_mode_(queue_mode) {}

  std::tuple<bool, unsigned int, std::vector<std::shared_ptr<PredictionData>>>
    add_prediction(unsigned int id, const VersionedModelId& model_id,
                  std::vector<std::shared_ptr<PredictionData>>&& predictions) {
      auto decoding_group_id = id / (ec_k_val_ + 1);
      auto decoding_group_it = decoding_groups_.find(decoding_group_id);
      if (decoding_group_it == decoding_groups_.end()) {
        log_debug_formatted(LOGGING_TAG_PARM, "Adding decoding group {} due to batch {}", decoding_group_id, id);
        decoding_group_it = decoding_groups_.emplace(
            decoding_group_id,
            DecodingGroup(decoding_group_id * (ec_k_val_ + 1), ec_k_val_ + 1)).first;
      }

      decoding_group_it->second.add_prediction(id, std::move(predictions));
      if (decoding_group_it->second.can_decode_data()) {
        log_debug_formatted(LOGGING_TAG_PARM, "Decoding batch {}", decoding_group_it->second.cur_missing_);
        auto start_time = std::chrono::system_clock::now();
        auto decoded = decode(decoding_group_it->second);
        auto end_time = std::chrono::system_clock::now();
        auto latency = end_time - start_time;
        long latency_micros = std::chrono::duration_cast<std::chrono::microseconds>(latency).count();
        log_info_formatted(LOGGING_TAG_EC_METRICS,
            "DECODING_LATENCY:group_id={},latency_micros={}",
            decoding_group_it->second.group_id_, latency_micros);

        // If the original task corresponding to the decoded task is still queued,
        // remove it from the queue.
        auto cur_missing = decoding_group_it->second.cur_missing_;
        long removed_queue_time = model_queues_->at(model_id)->remove_tasks_with_batch_id_lte(cur_missing);

        if (removed_queue_time >= 0) {
          log_debug_formatted(LOGGING_TAG_PARM, "Removing decoding group {}", decoding_group_id);
          decoding_groups_.erase(decoding_group_id);
        }
        return std::make_tuple(true, cur_missing, std::move(decoded));
      } else if (decoding_group_it->second.parity_useless()) {
        auto parity_id = decoding_group_it->second.parity_id_;
        long removed_queue_time = model_queues_->at(model_id)->remove_tasks_with_batch_id_lte(parity_id);
        if (removed_queue_time >= 0) {
          log_debug_formatted(LOGGING_TAG_PARM, "Removing decoding group {}", decoding_group_id);
          decoding_groups_.erase(decoding_group_id);
        }
      } else if (decoding_group_it->second.done()) {
        log_debug_formatted(LOGGING_TAG_PARM, "Removing decoding group {}", decoding_group_id);
        decoding_groups_.erase(decoding_group_id);
      }

      return std::make_tuple(false, 0, std::vector<std::shared_ptr<PredictionData>>());
    }

  virtual std::vector<std::shared_ptr<PredictionData>>
    decode(const DecodingGroup& decoding_group) = 0;

 protected:
  const size_t num_models_;
  const size_t num_red_models_;
  const size_t ec_k_val_;
  std::unordered_map<VersionedModelId, std::shared_ptr<ModelQueue>>* model_queues_;
  const QueueType queue_mode_;
  std::unordered_map<unsigned int, DecodingGroup> decoding_groups_;
};

class SubtractionDecoder : public Decoder {
 public:
  SubtractionDecoder(size_t num_models, size_t num_red_models, size_t ec_k_val,
                     std::unordered_map<VersionedModelId, std::shared_ptr<ModelQueue>>* model_queues,
                     QueueType queue_mode) :
    Decoder(num_models, num_red_models, ec_k_val, model_queues, queue_mode) {}

  virtual std::vector<std::shared_ptr<PredictionData>>
    decode(const DecodingGroup& dg) {
      std::vector<std::shared_ptr<PredictionData>> decoded_tasks;

      for (size_t i = 0; i < dg.decoding_buffer_[0].size(); i++) {
        auto decoded = dg.decoding_buffer_[dg.parity_pos_][i];
        auto decoded_data = get_data<float>(decoded).get();

        for (size_t j = 0; j < dg.decoding_buffer_.size(); j++) {
          if (j == dg.parity_pos_) {
            continue;
          }

          auto to_xor = dg.decoding_buffer_[j][i];
          auto to_xor_data = get_data<float>(to_xor).get();

          for (size_t b = 0; b < decoded->size() / sizeof(float); b++) {
            *(decoded_data + decoded->start() + b) -= *(to_xor_data + to_xor->start() + b);
          }
        } // for each prediction in batch
        decoded_tasks.emplace_back(decoded);
      } // for each batch in decoding group

      return std::move(decoded_tasks);
  }
};

}  // namespace clipper

#endif  // CLIPPER_LIB_DECODER_H
