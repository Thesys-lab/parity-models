#ifndef CLIPPER_LIB_TASK_EXECUTOR_H
#define CLIPPER_LIB_TASK_EXECUTOR_H

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <boost/optional.hpp>

#include <redox.hpp>

#include <folly/futures/Future.h>

#include <clipper/config.hpp>
#include <clipper/containers.hpp>
#include <clipper/decoder.hpp>
#include <clipper/datatypes.hpp>
#include <clipper/logging.hpp>
#include <clipper/metrics.hpp>
#include <clipper/model_queue.hpp>
#include <clipper/redis.hpp>
#include <clipper/rpc_service.hpp>
#include <clipper/threadpool.hpp>
#include <clipper/util.hpp>

namespace clipper {

const std::string LOGGING_TAG_TASK_EXECUTOR = "TASKEXECUTOR";

class ModelMetrics {
 public:
  explicit ModelMetrics(VersionedModelId model)
      : model_(model),
        latency_(metrics::MetricsRegistry::get_metrics().create_histogram(
            "model:" + model.serialize() + ":prediction_latency",
            "microseconds", 4096)),
        throughput_(metrics::MetricsRegistry::get_metrics().create_meter(
            "model:" + model.serialize() + ":prediction_throughput")),
        num_predictions_(metrics::MetricsRegistry::get_metrics().create_counter(
            "model:" + model.serialize() + ":num_predictions")),
        cache_hit_ratio_(
            metrics::MetricsRegistry::get_metrics().create_ratio_counter(
                "model:" + model.serialize() + ":cache_hit_ratio")),
        batch_size_(metrics::MetricsRegistry::get_metrics().create_histogram(
            "model:" + model.serialize() + ":batch_size", "queries", 4096)) {}
  ~ModelMetrics() = default;
  ModelMetrics(const ModelMetrics &) = default;
  ModelMetrics &operator=(const ModelMetrics &) = default;

  ModelMetrics(ModelMetrics &&) = default;
  ModelMetrics &operator=(ModelMetrics &&) = default;

  VersionedModelId model_;
  std::shared_ptr<metrics::Histogram> latency_;
  std::shared_ptr<metrics::Meter> throughput_;
  std::shared_ptr<metrics::Counter> num_predictions_;
  std::shared_ptr<metrics::RatioCounter> cache_hit_ratio_;
  std::shared_ptr<metrics::Histogram> batch_size_;
};

class CacheEntry {
 public:
  CacheEntry();
  ~CacheEntry() = default;

  CacheEntry(const CacheEntry &) = delete;
  CacheEntry &operator=(const CacheEntry &) = delete;

  CacheEntry(CacheEntry &&) = default;
  CacheEntry &operator=(CacheEntry &&) = default;

  bool completed_ = false;
  bool backup_completed_ = false;
  bool used_ = true;
  Output value_;
  Output backup_value_;
  std::vector<folly::Promise<Output>> value_promises_;
  std::vector<folly::Promise<Output>> backup_value_promises_;
};

// A cache page is a pair of <hash, entry_size>
using CachePage = std::pair<long, long>;

class PredictionCache {
 public:
  PredictionCache(size_t size_bytes);
  std::pair<folly::Future<Output>, folly::Future<Output>> fetch(PredictTask& t);

  void put(size_t cache_id, const Output &output, bool redundant);

 private:
  size_t hash(const VersionedModelId &model, size_t input_hash) const;
  void insert_entry(PredictTask&, CacheEntry &value);
  void evict_entries(long space_needed_bytes);

  std::mutex m_;
  const size_t max_size_bytes_;
  size_t size_bytes_ = 0;
  size_t cur_id_ = 0;
  size_t max_entries_ = 1000;
  std::queue<size_t> to_evict_list_;
  std::unordered_map<size_t, CacheEntry> entries_;
  std::vector<long> page_buffer_;
  size_t page_buffer_index_ = 0;
  std::shared_ptr<metrics::Counter> lookups_counter_;
  std::shared_ptr<metrics::RatioCounter> hit_ratio_;
};

struct DeadlineCompare {
  bool operator()(const std::pair<Deadline, PredictTask> &lhs,
                  const std::pair<Deadline, PredictTask> &rhs) {
    return lhs.first > rhs.first;
  }
};

class InflightMessage {
 public:
  InflightMessage(
      const std::chrono::time_point<std::chrono::system_clock> queue_time,
      const VersionedModelId model,
      const std::shared_ptr<PredictionData> input,
      const bool discard_result,
      const size_t cache_id)
      : queue_time_(std::move(queue_time)),
        sent_(false),
        model_(std::move(model)),
        input_(std::move(input)),
        discard_result_(discard_result),
        cache_id_(cache_id) {}

  void add_dispatch_info(
      const std::chrono::time_point<std::chrono::system_clock> send_time,
      const int container_id, const int replica_id) {
    send_time_ = std::move(send_time);
    sent_ = true;
    container_id_ = container_id;
    replica_id_ = replica_id;
  }

  // Default copy and move constructors
  InflightMessage(const InflightMessage &) = default;
  InflightMessage(InflightMessage &&) = default;

  // Default assignment operators
  InflightMessage &operator=(const InflightMessage &) = default;
  InflightMessage &operator=(InflightMessage &&) = default;

  std::chrono::time_point<std::chrono::system_clock> parity_time_;
  std::chrono::time_point<std::chrono::system_clock> queue_time_;
  std::chrono::time_point<std::chrono::system_clock> send_time_;
  bool sent_;
  int container_id_;
  VersionedModelId model_;
  int replica_id_;
  std::shared_ptr<PredictionData> input_;
  bool discard_result_;
  size_t cache_id_;
};

// Class for encompassing a batch of InflightMessages
class InflightMessageBatch {
 public:
   InflightMessageBatch() : redundant_(false) {}
   InflightMessageBatch(bool redundant, unsigned int batch_id) :
     redundant_(redundant),
     batch_id_(batch_id) {}

   std::vector<InflightMessage> messages_;
   bool redundant_;
   unsigned int batch_id_;
};

class TaskExecutor {
 public:
  ~TaskExecutor() { active_->store(false); };
  explicit TaskExecutor(RedundancyType redundancy_mode, QueueType queue_mode,
                        unsigned int num_models, unsigned int num_redundant_models,
                        unsigned int batch_size)
      : active_(std::make_shared<std::atomic_bool>(true)),
        active_containers_(std::make_shared<ActiveContainers>()),
        rpc_(std::make_unique<rpc::RPCService>()),
        cache_(std::make_unique<PredictionCache>(
            get_config().get_prediction_cache_size())),
        model_queues_({}),
        model_metrics_({}),
        redundancy_mode_(redundancy_mode),
        queue_mode_(queue_mode),
        num_models_(num_models),
        num_redundant_models_(num_redundant_models),
        batch_size_(batch_size) {
    log_debug(LOGGING_TAG_TASK_EXECUTOR, "TaskExecutor started");

    if (redundancy_mode_ == RedundancyType::NO_RED) {
      group_size_ = 1;
    } else if (redundancy_mode_ == RedundancyType::CHEAP) {
      group_size_ = 2;
    } else {
      size_t ec_k_val = num_models_ / num_redundant_models_;
      group_size_ = ec_k_val + 1;
    }

    decoder_.reset(
        new SubtractionDecoder(num_models, num_redundant_models_,
                               group_size_-1, &model_queues_, queue_mode_));
    rpc_->start(
        "*", RPC_SERVICE_PORT, [ this, task_executor_valid = active_ ](
                                   VersionedModelId model, int replica_id) {
          if (*task_executor_valid) {
            on_container_ready(model, replica_id);
          } else {
            log_debug(LOGGING_TAG_TASK_EXECUTOR,
                     "Not running on_container_ready callback because "
                     "TaskExecutor has been destroyed.");
          }
        },
        [ this, task_executor_valid = active_ ](rpc::RPCResponse & response) {
          if (*task_executor_valid) {
            on_response_recv(std::move(response));
          } else {
            log_debug(LOGGING_TAG_TASK_EXECUTOR,
                     "Not running on_response_recv callback because "
                     "TaskExecutor has been destroyed.");
          }

        },
        [ this, task_executor_valid = active_ ](VersionedModelId model,
                                                int replica_id) {
          if (*task_executor_valid) {
            on_remove_container(model, replica_id);
          } else {
            log_debug(LOGGING_TAG_TASK_EXECUTOR,
                     "Not running on_remove_container callback because "
                     "TaskExecutor has been destroyed.");
          }
        });
    Config &conf = get_config();
    while (!redis_connection_.connect(conf.get_redis_address(),
                                      conf.get_redis_port())) {
      log_error(LOGGING_TAG_TASK_EXECUTOR,
                "TaskExecutor failed to connect to redis",
                "Retrying in 1 second...");
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    while (!redis_subscriber_.connect(conf.get_redis_address(),
                                      conf.get_redis_port())) {
      log_error(LOGGING_TAG_TASK_EXECUTOR,
                "TaskExecutor subscriber failed to connect to redis",
                "Retrying in 1 second...");
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    redis::send_cmd_no_reply<std::string>(
        redis_connection_, {"CONFIG", "SET", "notify-keyspace-events", "AKE"});

    redis::subscribe_to_model_changes(redis_subscriber_, [
      this, task_executor_valid = active_
    ](const std::string &key, const std::string &event_type) {
      if (event_type == "hset" && *task_executor_valid) {
        auto model_info =
            clipper::redis::get_model_by_key(redis_connection_, key);
        VersionedModelId model_id = VersionedModelId(
            model_info["model_name"], model_info["model_version"]);
        int batch_size = DEFAULT_BATCH_SIZE;
        auto batch_size_search = model_info.find("batch_size");
        if (batch_size_search != model_info.end()) {
          batch_size = std::stoi(model_info["batch_size"]);
        }
        log_debug_formatted(LOGGING_TAG_TASK_EXECUTOR,
                           "Registered batch size of {} for model {}:{}",
                           batch_size, model_id.get_name(), model_id.get_id());
        active_containers_->register_batch_size(model_id, batch_size);
      }
    });

    redis::subscribe_to_container_changes(
        redis_subscriber_,
        // event_type corresponds to one of the Redis event types
        // documented in https://redis.io/topics/notifications.
        [ this, task_executor_valid = active_ ](const std::string &key,
                                                const std::string &event_type) {
          log_debug(LOGGING_TAG_TASK_EXECUTOR, "got subscribe_to_changes callback");
          if (event_type == "hset" && *task_executor_valid) {
            auto container_info =
                redis::get_container_by_key(redis_connection_, key);
            VersionedModelId vm = VersionedModelId(
                container_info["model_name"], container_info["model_version"]);
            unsigned int replica_id = std::stoi(container_info["model_replica_id"]);
            bool redundant = replica_id >= num_models_;
            active_containers_->add_container(
                vm, std::stoi(container_info["zmq_connection_id"]), replica_id,
                redundant, parse_input_type(container_info["input_type"]));

            auto model_info = redis::get_model(redis_connection_, vm);

            int batch_size = DEFAULT_BATCH_SIZE;
            auto batch_size_search = model_info.find("batch_size");
            if (batch_size_search != model_info.end()) {
              batch_size = std::stoi(batch_size_search->second);
            }
            active_containers_->register_batch_size(vm, batch_size);

            //EstimatorFittingThreadPool::create_queue(vm, replica_id);
            TaskExecutionThreadPool::create_queue(vm, replica_id);
            TaskExecutionThreadPool::submit_job(
                vm, replica_id, [this, vm, replica_id]() {
                  on_container_ready(vm, replica_id);
                });
            bool created_queue = create_model_queue_if_necessary(vm);
            if (created_queue) {
              log_debug_formatted(LOGGING_TAG_TASK_EXECUTOR,
                                 "Created queue for new model: {} : {}",
                                 vm.get_name(), vm.get_id());
            }
          } else if (!*task_executor_valid) {
            log_debug(LOGGING_TAG_TASK_EXECUTOR,
                     "Not running TaskExecutor's "
                     "subscribe_to_container_changes callback because "
                     "TaskExecutor has been destroyed.");
          }
        });
    throughput_meter_ = metrics::MetricsRegistry::get_metrics().create_meter(
        "internal:aggregate_model_throughput");
    predictions_counter_ =
        metrics::MetricsRegistry::get_metrics().create_counter(
            "internal:aggregate_num_predictions");
  }

  // Disallow copy
  TaskExecutor(const TaskExecutor &other) = delete;
  TaskExecutor &operator=(const TaskExecutor &other) = delete;

  TaskExecutor(TaskExecutor &&other) = default;
  TaskExecutor &operator=(TaskExecutor &&other) = default;

  typedef vector<folly::Future<Output>> OutputFutures;
  std::tuple<OutputFutures, OutputFutures> schedule_predictions(
      std::vector<PredictTask> tasks) {
    predictions_counter_->increment(tasks.size());
    std::vector<folly::Future<Output>> output_futures;
    std::vector<folly::Future<Output>> backup_output_futures;
    for (PredictTask& t : tasks) {
      // add each task to the queue corresponding to its associated model
      boost::shared_lock<boost::shared_mutex> lock(model_queues_mutex_);
      auto model_queue_entry = model_queues_.find(t.model_);

      if (model_queue_entry != model_queues_.end()) {
        auto cache_results = cache_->fetch(t);

        auto cache_result = std::move(cache_results.first);
        auto backup_cache_result = std::move(cache_results.second);

        if (cache_result.isReady()) {
          output_futures.push_back(std::move(cache_result));
          boost::shared_lock<boost::shared_mutex> model_metrics_lock(
              model_metrics_mutex_);
          auto cur_model_metric_entry = model_metrics_.find(t.model_);
          if (cur_model_metric_entry != model_metrics_.end()) {
            auto cur_model_metric = cur_model_metric_entry->second;
            cur_model_metric.cache_hit_ratio_->increment(1, 1);
          }
        }

        else if (active_containers_->get_replicas_for_model(t.model_).size() ==
                 0) {
          log_error_formatted(LOGGING_TAG_TASK_EXECUTOR,
                              "No active model containers for model: {} : {}",
                              t.model_.get_name(), t.model_.get_id());
        } else {
          output_futures.push_back(std::move(cache_result));

          t.recv_time_ = std::chrono::system_clock::now();
          log_debug_formatted(LOGGING_TAG_TASK_EXECUTOR,
                             "Adding task to queue. QueryID: {}, model: {}",
                             t.query_id_, t.model_.serialize());
          model_queue_entry->second->add_task(t, [this](Batch& batch) {
                auto queue_time = std::chrono::system_clock::now();
                InflightMessageBatch cur_batch(batch.redundant_, batch.batch_id_);

                for (auto b : batch.tasks_) {
                  cur_batch.messages_.emplace_back(b.recv_time_, b.model_, b.input_, b.artificial_, b.cache_id_);
                }

                if (batch.tasks_[0].is_parity_) {
                  for (size_t i = 0; i < cur_batch.messages_.size(); i++) {
                    cur_batch.messages_[i].parity_time_ = batch.tasks_[i].parity_time_;
                  }
                }

                inflight_messages_mutex_.lock();
                inflight_messages_.emplace(batch.batch_id_, cur_batch);
                inflight_messages_mutex_.unlock();
              });
          boost::shared_lock<boost::shared_mutex> model_metrics_lock(
              model_metrics_mutex_);
          auto cur_model_metric_entry = model_metrics_.find(t.model_);
          if (cur_model_metric_entry != model_metrics_.end()) {
            auto cur_model_metric = cur_model_metric_entry->second;
            cur_model_metric.cache_hit_ratio_->increment(0, 1);
          }
        }

        // We don't explicitly add a backup task here, but rather let
        // the underlying queue manage it.
        backup_output_futures.push_back(std::move(backup_cache_result));

      } else {
        log_debug_formatted(LOGGING_TAG_TASK_EXECUTOR,
                           "model_queues_.size() = {}",
                           model_queues_.size());
        for (auto kv : model_queues_)
          log_error_formatted(LOGGING_TAG_TASK_EXECUTOR,
                              "modle_queue contains key {} : {}",
                              kv.first.get_name(), kv.first.get_id());
        log_error_formatted(LOGGING_TAG_TASK_EXECUTOR,
                            "Received task for unknown model: {} : {}",
                            t.model_.get_name(), t.model_.get_id());
      }
    }
    return std::make_tuple(std::move(output_futures),
                           std::move(backup_output_futures));
  }

  std::vector<folly::Future<FeedbackAck>> schedule_feedback(
      const std::vector<FeedbackTask> tasks) {
    UNUSED(tasks);
    // TODO Implement
    return {};
  }

 private:
  // active_containers_ is shared with the RPC service so it can add new
  // containers to the collection when they connect
  std::shared_ptr<std::atomic_bool> active_;
  std::shared_ptr<ActiveContainers> active_containers_;
  std::unique_ptr<rpc::RPCService> rpc_;
  std::unique_ptr<PredictionCache> cache_;
  redox::Redox redis_connection_;
  redox::Subscriber redis_subscriber_;
  std::mutex inflight_messages_mutex_;
  std::unordered_map<unsigned int, InflightMessageBatch> inflight_messages_;
  std::unordered_map<int, unsigned int> rpc_id_to_batch_id_;
  std::shared_ptr<metrics::Counter> predictions_counter_;
  std::shared_ptr<metrics::Meter> throughput_meter_;
  boost::shared_mutex model_queues_mutex_;
  std::unordered_map<VersionedModelId, std::shared_ptr<ModelQueue>>
      model_queues_;
  boost::shared_mutex model_metrics_mutex_;
  std::unordered_map<VersionedModelId, ModelMetrics> model_metrics_;

  // Type of redundancy that should be used.
  const RedundancyType redundancy_mode_;

  // Load-balancing strategy that should be used.
  const QueueType queue_mode_;

  // Number of non-redundant models
  const unsigned int num_models_;

  // Number of batches in a group. Should be as follows:
  //   NO_RED: 1
  //   CHEAP: 2
  //   CODED: ec_k_ + 1
  // This should really be const, but my assignment pattern does not lend well
  // to the requirements of const variables.
  size_t group_size_;

  // Number of redundant models
  const unsigned int num_redundant_models_;

  // Batch size to be enforced
  const unsigned int batch_size_;

  std::unique_ptr<Decoder> decoder_;

  static constexpr int INITIAL_MODEL_QUEUES_MAP_SIZE = 100;

  bool create_model_queue_if_necessary(const VersionedModelId &model_id) {
    log_debug_formatted(LOGGING_TAG_PARM,
        "create_model_queue_if_necessary for {}:{}",
        model_id.get_name(), model_id.get_id());
    // Adds a new <model_id, task_queue> entry to the queues map, if one
    // does not already exist
    boost::unique_lock<boost::shared_mutex> l(model_queues_mutex_);
    bool queue_added = (model_queues_.find(model_id) == model_queues_.end());
    if (queue_added) {
      auto queue_emplace = model_queues_.emplace(
          std::make_pair(model_id, std::make_shared<ModelQueue>(
              redundancy_mode_, queue_mode_, num_models_,
              num_redundant_models_, batch_size_)));
      assert(queue_emplace.second);
      boost::unique_lock<boost::shared_mutex> l(model_metrics_mutex_);
      model_metrics_.insert(std::make_pair(model_id, ModelMetrics(model_id)));
    }
    return queue_added;
  }

  void on_container_ready(VersionedModelId model_id, int replica_id) {
    std::shared_ptr<ModelContainer> container =
        active_containers_->get_model_replica(model_id, replica_id);
    if (!container) {
      throw std::runtime_error(
          "TaskExecutor failed to find previously registered active "
          "container!");
    }
    boost::shared_lock<boost::shared_mutex> l(model_queues_mutex_);
    auto model_queue_entry = model_queues_.find(container->model_);
    if (model_queue_entry == model_queues_.end()) {
      throw std::runtime_error(
          "Failed to find model queue associated with a previously registered "
          "container!");
    }
    std::shared_ptr<ModelQueue> current_model_queue = model_queue_entry->second;
    // NOTE: It is safe to unlock here because we copy the shared_ptr to
    // the ModelQueue object so even if that entry in the map gets deleted,
    // the ModelQueue object won't be destroyed until our copy of the pointer
    // goes out of scope.
    l.unlock();

    Batch batch = current_model_queue->get_batch(container);

    if (batch.tasks_.size() > 0) {
      // move the lock up here, so that nothing can pull from the
      // inflight_messages_
      // map between the time a message is sent and when it gets inserted
      // into the map
      std::unique_lock<std::mutex> l(inflight_messages_mutex_);
      rpc::PredictionRequest prediction_request(container->input_type_);
      std::stringstream query_ids_in_batch;
      std::chrono::time_point<std::chrono::system_clock> current_time =
          std::chrono::system_clock::now();

      InflightMessageBatch& cur_batch = inflight_messages_[batch.batch_id_];
      for (size_t i = 0; i < cur_batch.messages_.size(); i++) {
        prediction_request.add_input(cur_batch.messages_[i].input_);
        cur_batch.messages_[i].add_dispatch_info(current_time, container->container_id_, container->replica_id_);
        query_ids_in_batch << batch.tasks_[i].query_id_ << " ";
      }
      int message_id = rpc_->send_message(prediction_request.serialize(),
                                          container->container_id_);
      log_debug_formatted(LOGGING_TAG_TASK_EXECUTOR,
                         "Sending batch to model: {} replica {}."
                         "Batch size: {}. Query IDs: {}",
                         model_id.serialize(), std::to_string(replica_id),
                         std::to_string(batch.tasks_.size()),
                         query_ids_in_batch.str());
      rpc_id_to_batch_id_.emplace(message_id, batch.batch_id_);
    } else {
      log_error_formatted(
          LOGGING_TAG_TASK_EXECUTOR,
          "ModelQueue returned empty batch for model {}, replica {}",
          model_id.serialize(), std::to_string(replica_id));
    }
  }

  void process_received(InflightMessageBatch& batch,
                        std::vector<std::shared_ptr<PredictionData>>& parsed_response_outputs,
                        bool is_reconstruction) {
    log_debug_formatted(LOGGING_TAG_PARM, "TaskExecutor::process_received batch_id={}, is_recon={}",
                       batch.batch_id_, is_reconstruction);
    size_t batch_size = batch.messages_.size();
    throughput_meter_->mark(batch_size);
    std::chrono::time_point<std::chrono::system_clock> current_time =
      std::chrono::system_clock::now();

    InflightMessage &first_message = batch.messages_[0];
    const VersionedModelId &cur_model = first_message.model_;
    boost::optional<ModelMetrics> cur_model_metric;
    auto cur_model_metric_entry = model_metrics_.find(cur_model);
    if (cur_model_metric_entry != model_metrics_.end()) {
      cur_model_metric = cur_model_metric_entry->second;
    }

    // Only do the following if not a reconstruction because we currently don't
    // have a good way of tracking model replicas in our redundancy implementation.
    if (!is_reconstruction) {
      const int cur_replica_id = first_message.replica_id_;
      auto batch_latency = current_time - first_message.send_time_;
      long long batch_latency_micros =
        std::chrono::duration_cast<std::chrono::microseconds>(batch_latency)
        .count();

      // Because an RPCResponse is guaranteed to contain data received from
      // a single model container, the processing container for the first
      // InflightMessage in the batch is the same processing container
      // for all InflightMessage objects in the batch
      std::shared_ptr<ModelContainer> processing_container =
        active_containers_->get_model_replica(cur_model, cur_replica_id);

      processing_container->add_processing_datapoint(batch_size,
          batch_latency_micros);

      if (cur_model_metric) {
        (*cur_model_metric).throughput_->mark(batch_size);
        (*cur_model_metric).num_predictions_->increment(batch_size);
        (*cur_model_metric).batch_size_->insert(batch_size);
      }
    }

    for (size_t batch_num = 0; batch_num < batch_size; ++batch_num) {
      InflightMessage completed_msg = batch.messages_[batch_num];
      if (!completed_msg.discard_result_) {
        cache_->put(completed_msg.cache_id_,
            Output{parsed_response_outputs[batch_num],
            {completed_msg.model_}}, is_reconstruction);
      }

      auto task_latency = current_time - completed_msg.queue_time_;
      long task_latency_micros =
        std::chrono::duration_cast<std::chrono::microseconds>(task_latency)
        .count();
      if (!is_reconstruction && cur_model_metric) {
        (*cur_model_metric)
          .latency_->insert(static_cast<int64_t>(task_latency_micros));
      }

      if (completed_msg.sent_) {
        auto queue_latency = completed_msg.send_time_ - completed_msg.queue_time_;
        long latency_micros =
          std::chrono::duration_cast<std::chrono::microseconds>(queue_latency).count();
        log_info_formatted(LOGGING_TAG_EC_METRICS,
                          "QUEUE_LATENCY:group_id={},batch_id={},query_id={},batch_size={},is_reconstruction={},latency_micros={}",
                          (batch.batch_id_ / group_size_), batch.batch_id_, batch_num,
                          batch_size, is_reconstruction, latency_micros);

        auto model_latency = current_time - completed_msg.send_time_;
        latency_micros =
          std::chrono::duration_cast<std::chrono::microseconds>(model_latency).count();
        log_debug_formatted(LOGGING_TAG_EC_METRICS,
                          "MODEL_LATENCY:group_id={},batch_id={},query_id={},batch_size={},is_reconstruction={},latency_micros={}",
                          (batch.batch_id_ / group_size_), batch.batch_id_, batch_num,
                          batch_size, is_reconstruction, latency_micros);
      }

      log_info_formatted(LOGGING_TAG_EC_METRICS,
                         "E2E_LATENCY:group_id={},batch_id={},query_id={},batch_size={},is_reconstruction={},latency_micros={}",
                         (batch.batch_id_ / group_size_), batch.batch_id_, batch_num,
                         batch_size, is_reconstruction, task_latency_micros);
    }
  }

  void on_response_recv(rpc::RPCResponse response) {
    std::unique_lock<std::mutex> l(inflight_messages_mutex_);
    log_debug_formatted(LOGGING_TAG_PARM, "Entered on_response_recv {}", response.first);
    auto if_id = rpc_id_to_batch_id_[response.first];
    auto batch = inflight_messages_[if_id];
    auto model_id = batch.messages_[0].model_;
    inflight_messages_.erase(if_id);
    rpc_id_to_batch_id_.erase(response.first);

    rpc::PredictionResponse parsed_response =
        rpc::PredictionResponse::deserialize_prediction_response(
            std::move(response.second));
    assert(parsed_response.outputs_.size() == batch.messages_.size());

    std::vector<std::shared_ptr<PredictionData>> copy_parsed_response_outputs;
    for (size_t i = 0; i < parsed_response.outputs_.size(); i++) {
      size_t size_bytes = parsed_response.outputs_[i]->byte_size();
      void* new_data = malloc(size_bytes);
      memcpy(new_data, (void*)get_data<uint8_t>(parsed_response.outputs_[i]).get(), size_bytes);
      UniquePoolPtr<void> data(new_data, free);
      copy_parsed_response_outputs.push_back(std::make_shared<ByteVector>(std::move(data), size_bytes));
    }

    // TODO: Probably can refactor to only hold this lock during process_received.
    // It currently will be held over both potential calls (this one and the potential
    // call from decoding).
    boost::shared_lock<boost::shared_mutex> metrics_lock(model_metrics_mutex_);
    if (!batch.redundant_ && batch.messages_.size() > 0) {
      process_received(batch, parsed_response.outputs_, false);

      if (redundancy_mode_ == RedundancyType::CHEAP) {
        // If the redundant batch has not yet been dequeued, remove it. The
        // redundant batch will have id batch_id_ + 1.
        model_queues_.at(model_id)->remove_tasks_with_batch_id_lte(batch.batch_id_ + 1);
      }
    }

    if (redundancy_mode_ == RedundancyType::CODED) {
      auto decode_or_none = decoder_->add_prediction(batch.batch_id_,
                                                     batch.messages_[0].model_,
                                                     std::move(copy_parsed_response_outputs));
      if (std::get<0>(decode_or_none) == true) {
        auto decoded_batch = inflight_messages_[std::get<1>(decode_or_none)];
        process_received(decoded_batch, std::get<2>(decode_or_none), true);
        // We do not delete the inflight message
      }
    } else if (redundancy_mode_ == RedundancyType::CHEAP) {
      if (batch.redundant_) {
        // If the "main" batch has not yet been dequeued, remove it. The
        // redundant batch will have id batch_id_ - 1.
        model_queues_.at(model_id)->remove_tasks_with_batch_id_lte(batch.batch_id_ - 1);

        // Decrement batch_id_ because this redundant batch is a response to
        // the queried batch with id batch_id_ - 1.
        batch.batch_id_--;
        process_received(batch, parsed_response.outputs_, true);
      }
    }

    if (batch.redundant_) {
      std::chrono::time_point<std::chrono::system_clock> current_time = std::chrono::system_clock::now();
      size_t batch_size = batch.messages_.size();
      for (size_t batch_num = 0; batch_num < batch_size; ++batch_num) {
        InflightMessage completed_msg = batch.messages_[batch_num];

        auto queue_latency = completed_msg.send_time_ - completed_msg.queue_time_;
        long latency_micros =
          std::chrono::duration_cast<std::chrono::microseconds>(queue_latency).count();
        log_info_formatted(LOGGING_TAG_EC_METRICS,
            "QUEUE_LATENCY_PARITY:group_id={},batch_id={},query_id={},batch_size={},latency_micros={}",
            (batch.batch_id_ / group_size_), batch.batch_id_, batch_num,
            batch_size, latency_micros);

        auto model_latency = current_time - completed_msg.send_time_;
        latency_micros =
          std::chrono::duration_cast<std::chrono::microseconds>(model_latency).count();
        log_debug_formatted(LOGGING_TAG_EC_METRICS,
            "MODEL_LATENCY_PARITY:group_id={},batch_id={},query_id={},batch_size={},latency_micros={}",
            (batch.batch_id_ / group_size_), batch.batch_id_, batch_num,
            batch_size, latency_micros);

        auto task_latency = current_time - completed_msg.parity_time_;
        long task_latency_micros =
          std::chrono::duration_cast<std::chrono::microseconds>(task_latency)
          .count();

        log_info_formatted(LOGGING_TAG_EC_METRICS,
            "E2E_LATENCY_PARITY:group_id={},batch_id={},query_id={},batch_size={},latency_micros={}",
            (batch.batch_id_ / group_size_), batch.batch_id_, batch_num,
            batch_size, task_latency_micros);
      }
    }
  }

  void on_remove_container(VersionedModelId model_id, int replica_id) {
    // remove the given model_id from active_containers_
    active_containers_->remove_container(model_id, replica_id);
  }
};

}  // namespace clipper

#endif  // CLIPPER_LIB_TASK_EXECUTOR_H
