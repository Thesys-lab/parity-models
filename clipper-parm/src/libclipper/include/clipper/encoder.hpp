#ifndef CLIPPER_LIB_ENCODER_H
#define CLIPPER_LIB_ENCODER_H

#include <chrono>
#include <future>
#include <thread>

#include <clipper/datatypes.hpp>
#include <clipper/logging.hpp>

#include <opencv2/opencv.hpp>

// Thanks: https://stackoverflow.com/questions/1486904/how-do-i-best-silence-a-warning-about-unused-variables
#define UNUSED(expr) do { (void)(expr); } while (0)

namespace clipper {

class Encoder {
 public:
   Encoder(unsigned int k, unsigned int batch_size) :
     ec_k_val_(k),
     batch_size_(batch_size) {}

   // Disallow copy and assign
   Encoder(const Encoder &) = delete;
   Encoder &operator=(const Encoder &) = delete;
   Encoder(Encoder &&) = delete;
   Encoder &operator=(Encoder &&) = delete;

   ~Encoder() = default;

   virtual void add_batch(Batch b) = 0;
   virtual bool parity_ready() = 0;
   virtual void add_batch_fn(Batch b, std::function<void(Batch&)>&& on_parity_add) {
     UNUSED(on_parity_add);
     log_info_formatted(LOGGING_TAG_PARM, "Unimplemented (satisfy compilier with this {})", b.batch_id_);
   }

   // Number of batches to wait for prior to encoding
   const unsigned int ec_k_val_;
   const unsigned int batch_size_;
};

class BatchEncoder : public Encoder {
 public:
  BatchEncoder(unsigned int k, unsigned int batch_size) :
    Encoder(k, batch_size) {}

  void add_batch(Batch b) {
    if (cur_group_.empty()) {
      start_group_ = b.tasks_[0].recv_time_;
    }
    cur_group_.push_back(std::move(b));
  }

  bool parity_ready() { return cur_group_.size() == ec_k_val_; }

 protected:
  std::vector<Batch> cur_group_;
  std::chrono::time_point<std::chrono::system_clock> start_group_;
};

class SingleBatchEncoderWorker {
 public:
  SingleBatchEncoderWorker(size_t ec_k_val) :
    ec_k_val_(ec_k_val),
    resize_len_(224),
    shift_(ec_k_val*0.485, ec_k_val*0.456, ec_k_val*0.406) {
    worker_lock_ = std::vector<std::mutex>(ec_k_val);
    worker_cv_ = std::vector<std::condition_variable>(ec_k_val);
    worker_go_ = std::vector<bool>(ec_k_val, false);
    worker_kill_ = std::vector<bool>(ec_k_val, false);
    for (size_t i = 0; i < ec_k_val; i++) {
      workers_.emplace_back(&SingleBatchEncoderWorker::worker_set_parity, this, i);
    }
  }

  ~SingleBatchEncoderWorker() {
    // NOTE: Had to add in destructor logic for cases where a ModelQueue is
    // spurriously destructed (this seems to happen early on in initialization).
    log_flush();
    for (size_t k = 0; k < ec_k_val_; k++) {
      worker_lock_[k].lock();
      worker_kill_[k] = true;
      worker_lock_[k].unlock();

      worker_cv_[k].notify_one();
      workers_[k].join();
    }
  }

  void load_and_convert_mat(const Batch& in_batch,
                            cv::Mat& decompressed_mat) {
    auto input = in_batch.tasks_[0].input_;
    uint8_t* in_data = get_data<uint8_t>(input).get();
    assert(in_data != nullptr);
    cv::Mat raw_data(1, input->size(), CV_8UC3, (void*)in_data);
    cv::Mat loaded_mat = cv::imdecode(raw_data, CV_LOAD_IMAGE_COLOR);
    if (loaded_mat.data == NULL) {
      log_error(LOGGING_TAG_PARM, "load_and_convert_mat_add() ERROR IN IMDECODE");
    }

    cv::Mat resized_mat(resize_len_, resize_len_, loaded_mat.type());
    cv::resize(loaded_mat, resized_mat, resized_mat.size());

    resized_mat.convertTo(decompressed_mat, CV_32FC3, 1.f/255);
  }

  void coordinator_set_parity(std::vector<Batch>* enc_group,
                              std::vector<PredictTask>& parity_tasks) {
    std::unique_lock<std::mutex> lock(coordinator_mutex_);
    cur_enc_group_ = enc_group;

    // Set up futures for the workers.
    future_mats_.clear();
    future_mat_promises_.clear();
    for (size_t k = 0; k < ec_k_val_; k++) {
      std::promise<cv::Mat> p;
      future_mats_.push_back(p.get_future());
      future_mat_promises_.push_back(std::move(p));
    }

    auto first_task = (*cur_enc_group_)[0].tasks_[0];
    size_t batch_id = (*cur_enc_group_)[0].batch_id_;

    // Set up our count and start workers
    num_workers_remaining_ = ec_k_val_;
    for (size_t k = 0; k < ec_k_val_; k++) {
      worker_lock_[k].lock();
      worker_go_[k] = true;
      worker_lock_[k].unlock();
      worker_cv_[k].notify_one();
    }

    // Wait for workers to finish.
    while (num_workers_remaining_ > 0) {
      coordinator_cv_.wait(lock, [this]() {return num_workers_remaining_ == 0;});
    }

    for (size_t k = 0; k < ec_k_val_; k++) {
      assert(!worker_go_[k]);
    }

    cv::Mat parity_mat = future_mats_[0].get();
    auto sum_start = std::chrono::system_clock::now();
    for (size_t i = 1; i < future_mats_.size(); i++) {
      parity_mat += future_mats_[i].get();
    }
    auto sum_stop = std::chrono::system_clock::now();

    normalize(parity_mat);

    long sum_micros = std::chrono::duration_cast<std::chrono::microseconds>(
          sum_stop - sum_start).count();

    log_info_formatted(LOGGING_TAG_EC_METRICS,
        "ENCODING_LATENCY_SUM:group_id={},latency_micros={}",
        (batch_id / (ec_k_val_ + 1)), sum_micros);

    size_t alloc_size = sizeof(float) * 3 * resize_len_ * resize_len_;
    void* new_parity_data = malloc(alloc_size);
    memcpy(new_parity_data, (void*)parity_mat.data, alloc_size);
    UniquePoolPtr<void> data(new_parity_data, free);
    std::shared_ptr<PredictionData> parity_task = std::make_shared<FloatVector>(std::move(data), alloc_size);
    parity_tasks.emplace_back(
        std::move(parity_task),
        first_task.model_, first_task.utility_, first_task.query_id_,
        first_task.latency_slo_micros_, first_task.artificial_);
  }

  void normalize(cv::Mat& in_mat) {
    in_mat -= shift_;
    std::vector<cv::Mat> bgr(3);
    cv::split(in_mat, bgr);
    bgr[0] *= 1.0/0.229;
    bgr[1] *= 1.0/0.224;
    bgr[2] *= 1.0/0.225;
    cv::merge(bgr, in_mat);
  }

  void worker_set_parity(size_t idx) {
    while (true) {
      std::unique_lock<std::mutex> lock(worker_lock_[idx]);
      while (!worker_kill_[idx] && !worker_go_[idx]) {
        worker_cv_[idx].wait(lock, [this, idx]() {return worker_kill_[idx] || worker_go_[idx];});
      }

      if (worker_kill_[idx]) {
        return;
      }

      cv::Mat decompressed_mat;
      load_and_convert_mat((*cur_enc_group_)[idx], decompressed_mat);

      coordinator_mutex_.lock();
      worker_go_[idx] = false;
      future_mat_promises_[idx].set_value(std::move(decompressed_mat));
      num_workers_remaining_ -= 1;
      coordinator_mutex_.unlock();
      coordinator_cv_.notify_one();
    }
  }

 private:
  const size_t ec_k_val_;
  const size_t resize_len_;
  cv::Vec3f shift_;

  std::vector<std::thread> workers_;
  std::vector<std::mutex> worker_lock_;
  std::vector<std::condition_variable> worker_cv_;

  // Signals for each worker on whether they have work to do.
  // Must be protected the corresponding worker_lock.
  std::vector<bool> worker_go_;

  // Signals each worker to exit their thread.
  // Must be protected the corresponding worker_lock.
  std::vector<bool> worker_kill_;

  // Needs to cover all that follow.
  std::mutex coordinator_mutex_;
  std::condition_variable coordinator_cv_;
  size_t num_workers_remaining_;
  std::vector<std::promise<cv::Mat>> future_mat_promises_;
  std::vector<std::future<cv::Mat>> future_mats_;
  std::vector<Batch>* cur_enc_group_;
};


class FloatEncoderQueueParallel : public BatchEncoder {
 public:
  // The values for normalization are taken from those used for ImageNet
  // training in the PyTorch example:
  //   https://github.com/pytorch/examples/tree/master/imagenet
  FloatEncoderQueueParallel(unsigned int k, unsigned int batch_size,
                         unsigned int num_redundant_workers) :
    BatchEncoder(k, batch_size),
    normalize_shift_(k*0.485, k*0.456, k*0.406),
    normalize_scale_(1.0/0.229, 1.0/0.224, 1.0/0.225),
    resize_len_(224),
    kill_encoder_threads_(false) {

    log_info_formatted(LOGGING_TAG_PARM, "FloatEncoderQueueParallel::constructor num_redundant_workers={}", num_redundant_workers);
    size_t num_red_workers_launch;
    if (num_redundant_workers >= 2) {
      num_red_workers_launch = num_redundant_workers / 2;
    } else {
      num_red_workers_launch = 1;
    }
    for (size_t i = 0; i < num_red_workers_launch; i++) {
      encoder_workers_.emplace_back(&FloatEncoderQueueParallel::worker_get_parity, this);
    }
    log_info_formatted(LOGGING_TAG_PARM, "FloatEncoderQueueParallel::constructor encoder_workers_.size()={}", encoder_workers_.size());
  }

  ~FloatEncoderQueueParallel() {
    // NOTE: Had to add in destructor logic for cases where a ModelQueue is
    // spurriously destructed (this seems to happen early on in initialization).
    encoder_queue_mutex_.lock();
    kill_encoder_threads_ = true;
    encoder_queue_mutex_.unlock();
    encoder_queue_cv_.notify_all();
    for (size_t i = 0; i < encoder_workers_.size(); i++) {
      encoder_workers_[i].join();
    }
  }

  virtual void add_batch(Batch b) {
    log_info_formatted(LOGGING_TAG_PARM, "Unimplemented (satisfy compiler by calling this {}", b.batch_id_);
  }

  void add_batch_fn(Batch b, std::function<void(Batch&)>&& on_parity_add) {
    std::lock_guard<std::mutex> overall_lock(encoder_mutex_);
    cur_group_.push_back(std::move(b));

    if (cur_group_.size() == ec_k_val_) {
      encoder_queue_mutex_.lock();
      encoder_queue_.emplace(std::move(cur_group_), on_parity_add);
      encoder_queue_mutex_.unlock();
      encoder_queue_cv_.notify_one();
      cur_group_ = std::vector<Batch>();
    }
  }

  void worker_get_parity() {
    SingleBatchEncoderWorker sub_worker(ec_k_val_);

    while (true) {
      // Make sure nothing weird happens where we try to grab a lock after
      // it has already been destroyed.
      if (kill_encoder_threads_) {
        return;
      }

      std::unique_lock<std::mutex> lock(encoder_queue_mutex_);
      while (!kill_encoder_threads_ && encoder_queue_.empty()) {
        encoder_queue_cv_.wait(lock, [this]() {return kill_encoder_threads_ || !encoder_queue_.empty();});
      }

      if (kill_encoder_threads_) {
        lock.unlock();
        encoder_queue_cv_.notify_all();
        return;
      }

      auto work_item = encoder_queue_.front();
      auto enc_group = work_item.first;
      auto on_batch_add = work_item.second;
      encoder_queue_.pop();

      lock.unlock();
      if (!encoder_queue_.empty()) {
        encoder_queue_cv_.notify_one();
      }

      // Batch id of the parity batch is one more than the final batch
      // in the queue.
      auto batch_id = enc_group.back().batch_id_ + 1;

      auto start_time = std::chrono::system_clock::now();
      auto start_group = enc_group[0].tasks_[0].recv_time_;
      auto latency = start_time - start_group;
      long latency_micros =
        std::chrono::duration_cast<std::chrono::microseconds>(latency)
        .count();
      log_info_formatted(LOGGING_TAG_EC_METRICS,
                         "ENCODING_GROUP_WAIT_LATENCY:group_id={},latency_micros={}",
                         (batch_id / (ec_k_val_ + 1)), latency_micros);

      // Note first task so that we can record its deadline, etc.
      auto first_batch = enc_group[0];

      std::vector<PredictTask> parity_tasks;
      parity_tasks.reserve(batch_size_);
      sub_worker.coordinator_set_parity(&enc_group, parity_tasks);

      auto end_time = std::chrono::system_clock::now();
      latency = end_time - start_time;
      latency_micros =
        std::chrono::duration_cast<std::chrono::microseconds>(latency)
        .count();
      log_info_formatted(LOGGING_TAG_EC_METRICS,
                         "ENCODING_LATENCY:group_id={},latency_micros={}",
                         (batch_id / (ec_k_val_ + 1)), latency_micros);

      for (size_t i = 0; i < parity_tasks.size(); i++) {
        parity_tasks[i].recv_time_ = end_time;
        parity_tasks[i].is_parity_ = true;
        parity_tasks[i].parity_time_ = start_group;
      }

      auto parity_batch = Batch(batch_id, first_batch.deadline_, std::move(parity_tasks));
      on_batch_add(parity_batch);
    }
  }

 private:
  // Vector used for image normalization
  cv::Vec3f normalize_shift_;
  cv::Vec3f normalize_scale_;
  size_t resize_len_;

  void normalize(cv::Mat& in_mat) {
    in_mat -= normalize_shift_;
    std::vector<cv::Mat> bgr(3);
    cv::split(in_mat, bgr);
    bgr[0] *= normalize_scale_[0];
    bgr[1] *= normalize_scale_[1];
    bgr[2] *= normalize_scale_[2];
    cv::merge(bgr, in_mat);
  }

  std::mutex encoder_mutex_;

  bool kill_encoder_threads_;
  std::queue<std::pair<std::vector<Batch>, std::function<void(Batch&)>>> encoder_queue_;
  std::vector<std::thread> encoder_workers_;
  std::mutex encoder_queue_mutex_;
  std::condition_variable encoder_queue_cv_;
};


class EncoderWorker {
 public:
  EncoderWorker(size_t ec_k_val, size_t batch_size) :
    ec_k_val_(ec_k_val),
    batch_size_(batch_size),
    resize_len_(224),
    shift_(ec_k_val*0.485, ec_k_val*0.456, ec_k_val*0.406) {
    worker_lock_ = std::vector<std::mutex>(batch_size);
    worker_cv_ = std::vector<std::condition_variable>(batch_size);
    worker_go_ = std::vector<bool>(batch_size, false);
    worker_kill_ = std::vector<bool>(batch_size, false);
    for (size_t i = 0; i < batch_size; i++) {
      workers_.emplace_back(&EncoderWorker::worker_set_parity, this, i);
    }
  }

  ~EncoderWorker() {
    // NOTE: Had to add in destructor logic for cases where a ModelQueue is
    // spurriously destructed (this seems to happen early on in initialization).
    log_flush();
    for (size_t b = 0; b < batch_size_; b++) {
      worker_lock_[b].lock();
      worker_kill_[b] = true;
      worker_lock_[b].unlock();

      worker_cv_[b].notify_one();
      workers_[b].join();
    }
  }

  void load_and_convert_mat_add(const Batch& in_batch,
                                size_t query_idx,
                                cv::Mat& cur_parity_mat,
                                bool first=false) {
    auto input = in_batch.tasks_[query_idx].input_;
    uint8_t* in_data = get_data<uint8_t>(input).get();
    assert(in_data != nullptr);
    cv::Mat raw_data(1, input->size(), CV_8UC3, (void*)in_data);
    cv::Mat loaded_mat = cv::imdecode(raw_data, CV_LOAD_IMAGE_COLOR);
    if (loaded_mat.data == NULL) {
      log_error_formatted(LOGGING_TAG_PARM, "load_and_convert_mat_add({}) ERROR IN IMDECODE", query_idx);
    }

    cv::Mat resized_mat(resize_len_, resize_len_, loaded_mat.type());
    cv::resize(loaded_mat, resized_mat, resized_mat.size());
    cv::Mat convert_mat;
    resized_mat.convertTo(convert_mat, CV_32FC3, 1.f/255);

    if (first) {
      cur_parity_mat = std::move(convert_mat);
    } else {
      cur_parity_mat += convert_mat;
    }
  }

  void coordinator_set_parity(std::vector<Batch>* enc_group,
                              std::vector<PredictTask>& parity_tasks) {
    std::unique_lock<std::mutex> lock(coordinator_mutex_);
    cur_enc_group_ = enc_group;

    // Set up futures for the workers.
    future_parity_tasks_.clear();
    future_parity_promises_.clear();
    for (size_t b = 0; b < batch_size_; b++) {
      std::promise<std::shared_ptr<PredictionData>> p;
      future_parity_tasks_.push_back(p.get_future());
      future_parity_promises_.push_back(std::move(p));
    }

    auto first_task = (*cur_enc_group_)[0].tasks_[0];

    // Set up our count and start workers
    num_workers_remaining_ = batch_size_;
    for (size_t b = 0; b < batch_size_; b++) {
      worker_lock_[b].lock();
      worker_go_[b] = true;
      worker_lock_[b].unlock();
      worker_cv_[b].notify_one();
    }

    // Wait for workers to finish.
    while (num_workers_remaining_ > 0) {
      coordinator_cv_.wait(lock, [this]() {return num_workers_remaining_ == 0;});
    }

    for (size_t b = 0; b < batch_size_; b++) {
      assert(!worker_go_[b]);
    }

    // Gather results
    for (size_t i = 0; i < future_parity_tasks_.size(); i++) {
      parity_tasks.emplace_back(future_parity_tasks_[i].get(), first_task.model_, first_task.utility_,
          first_task.query_id_, first_task.latency_slo_micros_, first_task.artificial_);
    }
  }

  void normalize(cv::Mat& in_mat) {
    in_mat -= shift_;
    std::vector<cv::Mat> bgr(3);
    cv::split(in_mat, bgr);
    bgr[0] *= 1.0/0.229;
    bgr[1] *= 1.0/0.224;
    bgr[2] *= 1.0/0.225;
    cv::merge(bgr, in_mat);
  }

  std::shared_ptr<FloatVector> debug_set_parity(size_t idx) {
    UNUSED(idx);
    size_t alloc_size = sizeof(float) * 3 * resize_len_ * resize_len_;
    void* new_parity_data = malloc(alloc_size);
    UniquePoolPtr<void> data(new_parity_data, free);
    return std::make_shared<FloatVector>(std::move(data), alloc_size);
  }

  void debug_coordinator_set_parity(std::vector<Batch>* enc_group,
                                    std::vector<PredictTask>& parity_tasks) {
    std::unique_lock<std::mutex> lock(coordinator_mutex_);
    cur_enc_group_ = enc_group;

    // Set up futures for the workers.
    future_parity_tasks_.clear();
    future_parity_promises_.clear();
    for (size_t b = 0; b < batch_size_; b++) {
      std::promise<std::shared_ptr<PredictionData>> p;
      future_parity_tasks_.push_back(p.get_future());
      future_parity_promises_.push_back(std::move(p));
    }


    for (size_t b = 0; b < batch_size_; b++) {
      future_parity_promises_[b].set_value(std::move(debug_set_parity(b)));
    }

    auto first_task = (*cur_enc_group_)[0].tasks_[0];

    // Gather results
    for (size_t i = 0; i < future_parity_tasks_.size(); i++) {
      parity_tasks.emplace_back(future_parity_tasks_[i].get(), first_task.model_, first_task.utility_,
          first_task.query_id_, first_task.latency_slo_micros_, first_task.artificial_);
    }
  }

  void worker_set_parity(size_t idx) {
    while (true) {
      std::unique_lock<std::mutex> lock(worker_lock_[idx]);
      while (!worker_kill_[idx] && !worker_go_[idx]) {
        worker_cv_[idx].wait(lock, [this, idx]() {return worker_kill_[idx] || worker_go_[idx];});
      }

      if (worker_kill_[idx]) {
        log_flush();
        return;
      }

      cv::Mat parity_mat;
      for (size_t k = 0; k < ec_k_val_; k++) {
        load_and_convert_mat_add((*cur_enc_group_)[k], idx, parity_mat, k == 0);
      }

      normalize(parity_mat);


      size_t alloc_size = sizeof(float) * 3 * resize_len_ * resize_len_;
      void* new_parity_data = malloc(alloc_size);
      memcpy(new_parity_data, (void*)parity_mat.data, alloc_size);
      UniquePoolPtr<void> data(new_parity_data, free);

      coordinator_mutex_.lock();
      worker_go_[idx] = false;
      future_parity_promises_[idx].set_value(std::make_shared<FloatVector>(std::move(data), alloc_size));
      num_workers_remaining_ -= 1;
      coordinator_mutex_.unlock();
      coordinator_cv_.notify_one();
    }
  }

 private:
  const size_t ec_k_val_;
  const size_t batch_size_;
  const size_t resize_len_;
  cv::Vec3f shift_;

  std::vector<std::thread> workers_;
  std::vector<std::mutex> worker_lock_;
  std::vector<std::condition_variable> worker_cv_;

  // Signals for each worker on whether they have work to do.
  // Must be protected the corresponding worker_lock.
  std::vector<bool> worker_go_;

  // Signals each worker to exit their thread.
  // Must be protected the corresponding worker_lock.
  std::vector<bool> worker_kill_;

  // Needs to cover all that follow.
  std::mutex coordinator_mutex_;
  std::condition_variable coordinator_cv_;
  size_t num_workers_remaining_;
  std::vector<std::promise<std::shared_ptr<PredictionData>>> future_parity_promises_;
  std::vector<std::future<std::shared_ptr<PredictionData>>> future_parity_tasks_;
  std::vector<Batch>* cur_enc_group_;
};

class BatchFloatEncoderQueueWorkers : public BatchEncoder {
 public:
  BatchFloatEncoderQueueWorkers(unsigned int k, unsigned int batch_size,
                                unsigned int num_redundant_workers) :
    BatchEncoder(k, batch_size),
    kill_encoder_threads_(false) {

    size_t num_workers = num_redundant_workers / 2;
    if (num_workers == 0) {
      num_workers = 1;
    }
    log_debug_formatted(LOGGING_TAG_PARM, "BatchFloatEncoderQueueWorkers::constructor num_redundant_workers={}", num_workers);
    for (size_t i = 0; i < num_workers; i++) {
      encoder_workers_.emplace_back(&BatchFloatEncoderQueueWorkers::worker_get_parity, this);
    }
    log_debug_formatted(LOGGING_TAG_PARM, "BatchFloatEncoderQueueWorkers::constructor encoder_workers_.size()={}", encoder_workers_.size());
  }

  ~BatchFloatEncoderQueueWorkers() {
    // NOTE: Had to add in destructor logic for cases where a ModelQueue is
    // spurriously destructed (this seems to happen early on in initialization).
    log_flush();
    encoder_queue_mutex_.lock();
    kill_encoder_threads_ = true;
    encoder_queue_mutex_.unlock(); encoder_queue_cv_.notify_all();
    for (size_t i = 0; i < encoder_workers_.size(); i++) {
      encoder_workers_[i].join();
    }
  }

  virtual void add_batch(Batch b) {
    log_info_formatted(LOGGING_TAG_PARM, "Unimplemented (satisfy compiler by calling this {}", b.batch_id_);
  }

  void add_batch_fn(Batch b, std::function<void(Batch&)>&& on_parity_add) {
    std::lock_guard<std::mutex> overall_lock(encoder_mutex_);
    cur_group_.push_back(std::move(b));

    if (cur_group_.size() == ec_k_val_) {
      encoder_queue_mutex_.lock();
      encoder_queue_.emplace(std::move(cur_group_), on_parity_add);
      encoder_queue_mutex_.unlock();
      encoder_queue_cv_.notify_one();
      cur_group_ = std::vector<Batch>();
    }
  }

  void worker_get_parity() {
    EncoderWorker sub_worker(ec_k_val_, batch_size_);
    while (true) {
      // Make sure nothing weird happens where we try to grab a lock after
      // it has already been destroyed.
      if (kill_encoder_threads_) {
        return;
      }

      std::unique_lock<std::mutex> lock(encoder_queue_mutex_);
      while (!kill_encoder_threads_ && encoder_queue_.empty()) {
        encoder_queue_cv_.wait(lock, [this]() {return kill_encoder_threads_ || !encoder_queue_.empty();});
      }

      if (kill_encoder_threads_) {
        lock.unlock();
        encoder_queue_cv_.notify_all();
        return;
      }

      auto work_item = encoder_queue_.front();
      auto enc_group = work_item.first;
      auto on_batch_add = work_item.second;
      encoder_queue_.pop();

      lock.unlock();
      if (!encoder_queue_.empty()) {
        encoder_queue_cv_.notify_one();
      }

      // Batch id of the parity batch is one more than the final batch
      // in the queue.
      auto batch_id = enc_group.back().batch_id_ + 1;
      auto first_batch_deadline = enc_group[0].deadline_;

      auto start_time = std::chrono::system_clock::now();
      auto start_group = enc_group[0].tasks_[0].recv_time_;
      auto latency = start_time - start_group;
      long latency_micros =
        std::chrono::duration_cast<std::chrono::microseconds>(latency)
        .count();
      log_info_formatted(LOGGING_TAG_EC_METRICS,
                         "ENCODING_GROUP_WAIT_LATENCY:group_id={},latency_micros={}",
                         (batch_id / (ec_k_val_ + 1)), latency_micros);

      std::vector<PredictTask> parity_tasks;
      parity_tasks.reserve(batch_size_);
      sub_worker.coordinator_set_parity(&enc_group, parity_tasks);

      auto end_time = std::chrono::system_clock::now();
      latency = end_time - start_time;
      latency_micros =
        std::chrono::duration_cast<std::chrono::microseconds>(latency)
        .count();
      log_info_formatted(LOGGING_TAG_EC_METRICS,
                         "ENCODING_LATENCY:group_id={},latency_micros={}",
                         (batch_id / (ec_k_val_ + 1)), latency_micros);

      for (size_t i = 0; i < parity_tasks.size(); i++) {
        parity_tasks[i].recv_time_ = end_time;
        parity_tasks[i].is_parity_ = true;
        parity_tasks[i].parity_time_ = start_group;
      }

      auto parity_batch = Batch(batch_id, first_batch_deadline, std::move(parity_tasks));
      on_batch_add(parity_batch);
    }
  }

 private:
  std::mutex encoder_mutex_;
  bool kill_encoder_threads_;
  std::queue<std::pair<std::vector<Batch>, std::function<void(Batch&)>>> encoder_queue_;
  std::vector<std::thread> encoder_workers_;
  std::mutex encoder_queue_mutex_;
  std::condition_variable encoder_queue_cv_;
};


}  // namespace clipper

#endif  // CLIPPER_LIB_ENCODER_H
