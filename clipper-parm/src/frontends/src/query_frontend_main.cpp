#include <clipper/config.hpp>
#include <clipper/constants.hpp>
#include <clipper/datatypes.hpp>
#include <clipper/logging.hpp>
#include <clipper/query_processor.hpp>
#include <cxxopts.hpp>

#include "query_frontend.hpp"

int main(int argc, char* argv[]) {
#ifndef NDEBUG
  spdlog::set_level(spdlog::level::debug);
#else
  spdlog::set_level(spdlog::level::info);
#endif

  cxxopts::Options options("query_frontend",
                           "Clipper query processing frontend");
  // clang-format off
  options.add_options()
    ("redis_ip", "Redis address",
        cxxopts::value<std::string>()->default_value(clipper::DEFAULT_REDIS_ADDRESS))
    ("redis_port", "Redis port",
        cxxopts::value<int>()->default_value(std::to_string(clipper::DEFAULT_REDIS_PORT)))
    ("prediction_cache_size", "Size of the prediction cache in bytes, excluding cache metadata",
       cxxopts::value<long>()->default_value(std::to_string(clipper::DEFAULT_PREDICTION_CACHE_SIZE_BYTES)))
    ("redundancy_mode", "0: NO_RED, 1: REPLICATION, 2: CODED",
       cxxopts::value<unsigned int>()->default_value(std::to_string(0)))
    ("queue_mode", "0: SINGLE, 1: ROUND_ROBIN",
       cxxopts::value<unsigned int>()->default_value(std::to_string(0)))
    ("num_models", "Number of replicas of the original task that will be made",
       cxxopts::value<unsigned int>()->default_value(std::to_string(1)))
    ("num_redundant_models", "Number of redundant tasks that will be deployed",
       cxxopts::value<unsigned int>()->default_value(std::to_string(1)))
    ("batch_size", "Batch size to enforce",
       cxxopts::value<unsigned int>()->default_value(std::to_string(1)));
  // clang-format on
  options.parse(argc, argv);
  clipper::Config& conf = clipper::get_config();
  conf.set_redis_address(options["redis_ip"].as<std::string>());
  conf.set_redis_port(options["redis_port"].as<int>());
  conf.set_prediction_cache_size(options["prediction_cache_size"].as<long>());
  conf.ready();

  clipper::RedundancyType redundancy_mode = static_cast<clipper::RedundancyType>(options["redundancy_mode"].as<unsigned int>());
  clipper::QueueType queue_mode = static_cast<clipper::QueueType>(options["queue_mode"].as<unsigned int>());
  unsigned int num_models = options["num_models"].as<unsigned int>();
  unsigned int num_redundant_models = options["num_redundant_models"].as<unsigned int>();
  unsigned int batch_size = options["batch_size"].as<unsigned int>();
  clipper::log_info_formatted(clipper::LOGGING_TAG_PARM,
      "Options are: red={}, queue={}, num_models={}, num_red={}, batch_size={}",
      static_cast<int>(redundancy_mode), static_cast<int>(queue_mode), num_models, num_redundant_models, batch_size);

  query_frontend::RequestHandler<clipper::QueryProcessor> rh(
      "0.0.0.0", clipper::QUERY_FRONTEND_PORT, redundancy_mode, queue_mode,
      num_models, num_redundant_models, batch_size);
  rh.start_listening();
}
