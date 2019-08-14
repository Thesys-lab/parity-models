#ifndef CLIPPER_LIB_MODEL_QUEUE_H
#define CLIPPER_LIB_MODEL_QUEUE_H

#include <chrono>

#include <clipper/datatypes.hpp>
#include <clipper/encoder.hpp>
#include <clipper/logging.hpp>

#include <thread>

namespace clipper {

// thread safe model queue
class ModelQueue {
 public:
  ModelQueue(RedundancyType redundancy_mode, QueueType queue_mode,
             unsigned int num_models, unsigned int num_redundant_models,
             unsigned int batch_size) :
    batch_id_(0),
    redundancy_mode_(redundancy_mode),
    queue_mode_(queue_mode),
    num_models_(num_models),
    num_redundant_models_(num_redundant_models),
    batch_size_(batch_size),
    num_batches_backup_(1),
    cur_queue_(0),
    cur_parity_queue_add_(0),
    encoder_(nullptr) {

      if (redundancy_mode_ == RedundancyType::NO_RED)
        assert(num_redundant_models_ == 0);

      int num_queues = num_models_;
      if (queue_mode_ == QueueType::SINGLE) {
        num_queues = 1;
      }

      if (redundancy_mode_ == RedundancyType::CODED ||
          redundancy_mode_ == RedundancyType::CHEAP) {
        if (queue_mode_ == QueueType::SINGLE) {
          // Use one more queue for parity queries.
          num_queues += 1;
        } else {
          // Use as many queues as there are parity models.
          num_queues += num_redundant_models;
        }
      }
      if (redundancy_mode_ == RedundancyType::CODED) {
        size_t ec_k_val = num_models / num_redundant_models;
        log_debug_formatted(LOGGING_TAG_PARM, "ec_k_val is {}", ec_k_val);

        if (batch_size == 1) {
          encoder_.reset(new FloatEncoderQueueParallel(ec_k_val, batch_size, num_redundant_models));
        } else {
          encoder_.reset(new BatchFloatEncoderQueueWorkers(ec_k_val, batch_size, num_redundant_models));
        }
      }

      log_debug_formatted(LOGGING_TAG_PARM,
                         "Starting ModelQueue with {} queues, num_models = {}, red={}, queue={}, num_red={}, batch_size={}",
                         num_queues, num_models_, static_cast<int>(redundancy_mode_), static_cast<int>(queue_mode_), num_redundant_models_, batch_size_);
      queue_ = std::vector<std::queue<Batch>>(num_queues);
      queue_mutex_ = std::vector<std::mutex>(num_queues);
      queue_cv_ = std::vector<std::condition_variable>(num_queues);
    }

  // Disallow copy and assign
  ModelQueue(const ModelQueue &) = delete;
  ModelQueue &operator=(const ModelQueue &) = delete;

  ModelQueue(ModelQueue &&) = default;
  ModelQueue &operator=(ModelQueue &&) = default;

  void add_task(PredictTask task,
                std::function<void(Batch&)>&& on_batch_add) {

    add_to_batch(std::move(task), std::move(on_batch_add));
  }

  void add_to_batch(PredictTask task,
                    std::function<void(Batch&)>&& on_batch_add) {
    std::lock_guard<std::mutex> overall_lock(dispatch_mutex_);

    cur_predictions_.push_back(std::move(task));

    if (cur_predictions_.size() == batch_size_) {
      Deadline deadline = std::chrono::system_clock::now() +
                          std::chrono::microseconds(cur_predictions_.back().latency_slo_micros_);

      size_t queue_idx;
      if (queue_mode_ == QueueType::SINGLE) {
        queue_idx = 0;
      } else {
        queue_idx = cur_queue_;
        cur_queue_ = (cur_queue_ + 1) % num_models_;
      }

      Batch batch_og = Batch(batch_id_, deadline, std::move(cur_predictions_));
      Batch batch_rep(batch_og);

      batch_id_to_queue_id_mutex_.lock();
      batch_id_to_queue_id_.emplace(batch_id_, queue_idx);
      batch_id_to_queue_id_mutex_.unlock();
      batch_id_++;

      on_batch_add(batch_og);
      queue_mutex_[queue_idx].lock();
      log_debug_formatted(LOGGING_TAG_PARM, "Adding batch to queue {}", queue_idx);
      queue_[queue_idx].push(std::move(batch_og));
      queue_mutex_[queue_idx].unlock();
      queue_cv_[queue_idx].notify_one();

      // Perform redundancy logic.
      if (redundancy_mode_ == RedundancyType::CHEAP) {

        // Determine the parity queue. This will depend on both queue_mode_
        // and redundancy_mode_.
        size_t parity_queue;
        if (queue_mode_ == QueueType::SINGLE) {
          // For CHEAP, the redundant queue is separate from the "main" queue,
          parity_queue = 1;
        } else {
          // For CHEAP, we perform round-robin through the redundant queues.
          parity_queue = num_models_ + cur_parity_queue_add_;
          cur_parity_queue_add_ = (cur_parity_queue_add_ + 1) % num_redundant_models_;
        }

        // The redundant batch will receive a batch id one higher than the
        // "original" batch.
        batch_rep.redundant_ = true;
        batch_rep.batch_id_ = batch_id_;

        batch_id_to_queue_id_mutex_.lock();
        batch_id_to_queue_id_.emplace(batch_id_, parity_queue);
        batch_id_to_queue_id_mutex_.unlock();

        batch_id_++;

        log_debug_formatted(LOGGING_TAG_PARM, "Adding redundant batch to queue {}", parity_queue);
        on_batch_add(batch_rep);
        queue_mutex_[parity_queue].lock();
        queue_[parity_queue].push(std::move(batch_rep));
        queue_mutex_[parity_queue].unlock();
        queue_cv_[parity_queue].notify_one();
      } else if (redundancy_mode_ == RedundancyType::CODED) {
        size_t parity_queue;

        // Check if adding this batch to the encoder would finish an
        // encoding group. This will be the case if the current batch_id_
        // (after having been incremented) modulo the encoding group size
        // (ec_k_val_ + 1) is equal to ec_k_val_.
        if ((batch_id_ % (encoder_->ec_k_val_ + 1)) == encoder_->ec_k_val_) {
          if (queue_mode_ == QueueType::SINGLE) {
            parity_queue = 1;
          } else {
            parity_queue = num_models_ + cur_parity_queue_add_;
            cur_parity_queue_add_ = (cur_parity_queue_add_ + 1) % num_redundant_models_;
          }

          batch_id_to_queue_id_mutex_.lock();
          batch_id_to_queue_id_.emplace(batch_id_, parity_queue);
          batch_id_to_queue_id_mutex_.unlock();

          // Increment the running batch_id_ again so as to account for the
          // batch_id_ which will be given to the parity batch.
          batch_id_++;
        } else {
          // As this batch won't finish off a parity group, its parity_queue
          // will not be used.
          parity_queue = UINT_MAX;
        }

        encoder_->add_batch_fn(std::move(batch_rep),
            [this, on_batch_add, parity_queue](Batch& batch) {
                batch.redundant_ = true;
                on_batch_add(batch);
                queue_mutex_[parity_queue].lock();
                queue_[parity_queue].push(std::move(batch));
                queue_mutex_[parity_queue].unlock();
                queue_cv_[parity_queue].notify_one();
              }
            );
      }

      // Clear the current predictions
      cur_predictions_ = std::vector<PredictTask>();
    }
  }

  Batch get_batch(std::shared_ptr<ModelContainer> requesting_container) {

    // If queue_mode is anything other than SINGLE, then containers pull from
    // their own individual queue. Otherwise, non-redundant containers pull
    // from the "main" queue and redundant containers pull from the redundant
    // queue.
    int queue_id = requesting_container->replica_id_;
    if (queue_mode_ == QueueType::SINGLE) {
      if (requesting_container->replica_id_ >= (int)num_models_) {
        queue_id = 1;
      } else {
        queue_id = 0;
      }
    }

    log_debug_formatted(LOGGING_TAG_PARM, "Container waiting from queue {}", queue_id);
    std::unique_lock<std::mutex> lock(queue_mutex_[queue_id]);
    remove_tasks_with_elapsed_deadlines(queue_id);
    while (queue_[queue_id].empty()) {
      queue_cv_[queue_id].wait(lock, [this, queue_id]() {return !queue_[queue_id].empty();});
      remove_tasks_with_elapsed_deadlines(queue_id);
    }

    Batch batch = queue_[queue_id].front();

    log_debug_formatted(LOGGING_TAG_PARM, "Container pulling {} from queue {}", batch.batch_id_, queue_id);
    queue_[queue_id].pop();
    lock.unlock();

    batch_id_to_queue_id_mutex_.lock();
    batch_id_to_queue_id_.erase(batch.batch_id_);
    batch_id_to_queue_id_mutex_.unlock();

    if (!queue_[queue_id].empty()) {
      queue_cv_[queue_id].notify_one();
    }

    return batch;
  }

  // Removes batches that have batch_id less than or equal to `max_batch_id`
  // from the queue that the batch with `max_batch_id` is on, if it is still
  // enqueued.
  long remove_tasks_with_batch_id_lte(unsigned int max_batch_id) {
    batch_id_to_queue_id_mutex_.lock();
    auto queue_id_it = batch_id_to_queue_id_.find(max_batch_id);

    if (queue_id_it == batch_id_to_queue_id_.end()) {
      batch_id_to_queue_id_mutex_.unlock();
      return -1;
    }

    size_t queue_id = queue_id_it->second;
    long ret_val = -1;
    queue_mutex_[queue_id].lock();
    std::chrono::time_point<std::chrono::system_clock> current_time =
        std::chrono::system_clock::now();
    while (!queue_[queue_id].empty()) {
      auto batch_id = queue_[queue_id].front().batch_id_;
      if (batch_id <= max_batch_id) {
        log_debug_formatted(LOGGING_TAG_PARM, "Removing batch {} from queue {}", batch_id, queue_id);

        for (size_t i = 0; i < queue_[queue_id].front().tasks_.size(); i++) {
          long latency_micros =
            std::chrono::duration_cast<std::chrono::microseconds>(current_time - queue_[queue_id].front().tasks_[i].recv_time_)
            .count();
          ret_val = latency_micros;
          log_debug_formatted(LOGGING_TAG_EC_METRICS, "QUEUE_LATENCY_REMOVED:group_id={},batch_id={},query_id={},latency_micros={}",
              (batch_id / (num_models_ + 1)), batch_id, i, latency_micros);
        }

        batch_id_to_queue_id_.erase(batch_id);
        queue_[queue_id].pop();
      } else {
        break;
      }
    }

    queue_mutex_[queue_id].unlock();
    batch_id_to_queue_id_mutex_.unlock();
    return ret_val;
  }

 private:
  std::vector<std::queue<Batch>> queue_;
  std::vector<std::mutex> queue_mutex_;
  std::vector<std::condition_variable> queue_cv_;

  // Must be held when updating batch_id_, cur_queue_,
  // cur_parity_queue_add_, and cur_predictions_.
  std::mutex dispatch_mutex_;

  unsigned int batch_id_;

  // Type of redundancy that should be used.
  const RedundancyType redundancy_mode_;

  // Load-balancing strategy that should be used.
  const QueueType queue_mode_;

  // Number of non-redundant models
  const unsigned int num_models_;

  // Number of redundant models
  const unsigned int num_redundant_models_;

  // Batch size to be enforced
  const unsigned int batch_size_;

  // How many batches should be queued in the backup before popping.
  const unsigned short num_batches_backup_;

  // Current queue to which queries should be dispatched.
  unsigned int cur_queue_;

  // Current index of parity queue. This is used in determining which queue
  // to dispatch a parity query to when using multiple queues.
  unsigned int cur_parity_queue_add_;

  // Current predictions which will be batched together and dispatched to a
  // model replica.
  std::vector<PredictTask> cur_predictions_;

  // Encoder to be used in case of redundancy_mode_ == CODED
  std::unique_ptr<Encoder> encoder_;

  // Keeps track of the queue to which each batch was placed.
  std::mutex batch_id_to_queue_id_mutex_;
  std::unordered_map<size_t, size_t> batch_id_to_queue_id_;

  // Deletes tasks with deadlines prior or equivalent to the
  // current system time. This method should only be called
  // when a unique lock on the queue_mutex is held.
  void remove_tasks_with_elapsed_deadlines(int queue_id) {
    std::chrono::time_point<std::chrono::system_clock> current_time =
        std::chrono::system_clock::now();
    while (!queue_[queue_id].empty()) {
      Deadline first_deadline = queue_[queue_id].front().deadline_;
      auto recv_time = queue_[queue_id].front().tasks_[0].recv_time_;
      if (first_deadline <= current_time) {
        // If a task's deadline has already elapsed,
        // we should not process it
        log_debug_formatted(LOGGING_TAG_PARM, "Removing batch from queue {}", queue_id);
        long latency_micros =
        std::chrono::duration_cast<std::chrono::microseconds>(current_time - recv_time)
        .count();
        log_debug_formatted(LOGGING_TAG_PARM, "Has been {} micros since received", latency_micros);
        queue_[queue_id].pop();
      } else {
        break;
      }
    }
  }
};

}  // namespace clipper

#endif  // CLIPPER_LIB_MODEL_QUEUE_H
