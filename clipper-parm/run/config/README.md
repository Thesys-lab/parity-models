# Latency configuration settings
This directory contains configuration files used in evaluating ParM.

## Parameters
* `experiment_id`: A tag that will be assigned to all AWS instances that are
run for this experiment. Experiments that are run using the same ID will
share the same tag. This makes it easy to reuse instances for one experiment
after a previous one has completed. If you would like to run multiple
experiments on disjoint sets of AWS instances, then you should consider using
a different `experiment_id` for each configuration. **NOTE:** using multiple
IDs will break the current instance termination scripts; you are advised to
manually terminate instances in this case.
* `frontend_type`: AWS instance type to be used for the frontend
* `worker_type`: AWS instance type to be used for workers
* `client_type`: AWS instance type to be used for clients
* `model`: Architecture for the base model and parity models
* `s3_path`: Deprecated
* `queue_modes`: The queueing strategy that will be used in this experiment.
Currently supported options include:
  - `single_queue`: The frontend maintains a single queue from which model
  instances pull queries.
  - `rr`: The frontend maintains a queue for each model instance, and queries
  are added to each queue in a round-robin fashion.
* `num_models`: The number of instances to be launched for performing inference
over copies of the base model.
* `ec_k_val`: The value of parameter k to be used. See `redundancy_modes` for
additional details.
* `redundancy_modes`: The type of redundancy to be used. Currently supported
options include:
  - `none`: No redundancy. Only `num_models` base models are launched.
  - `equal`: Uses `num_models / ec_k_val` additional model instances to serve
  extra copies of the base model.
  - `coded`: Uses `num_models / ec_k_val` additional model instances to serve
  parity models
  - `cheap`: Uses `num_models / ec_k_val` additional model instances ot serve
  models that approximate the base model. We currently use Mobilenet-V2 with a
  width factor of 0.25 or these models.
* `batch_sizes`: The number of queries that will be batched together before
being dispatched to a model instance.
* `num_clients`: The number of client instances to launch
* `send_rates`: The arrival rate of queries to the frontend. Clients will work
together to send queries following a Poisson arrival process with expected
arrival rate equal to the value set by this parameter.
* `background_traffic`: The amount and type of background load to be induced
on instances used in experiments. Currently supported options include:
  - Non-negative integer values: These specify to induce background network
  shuffles between model instances. The value specified indicates the number
  of concurrent background shuffles (0 means no background traffic, 4 means
  four concurrent background shuffles).
  - "clipper": Specifies to launch another Clipper frontend and to colocate
  some of the model instances associated with this frontend with model
  instances used by our main experimental instances. Details on this approach
  may be found in [launch_background.py](../launch_background.py)
* `num_queries`: The number of queries to be processed in an experiment
* `num_trials`: The number of times each experiment should be repeated

## Which configurations are run?
Many configuration
parameters are represented as lists (e.g., `send_rates`). All combinations of
configuration parameters are executed. For example, cosider a configuration
file with the following contents (found in [fig9a.json](fig9a.json)):
```
  "experiment_id": "ParM",
  "frontend_type": "c5.9xlarge",
  "worker_type": "p2.xlarge",
  "client_type": "m5.xlarge",
  "model": "resnet18",
  "s3_path": "",

  "queue_modes": ["single_queue"],
  "redundancy_modes": ["equal", "coded"],
  "num_models": 12,
  "ec_k_val": [2],
  "batch_sizes": [1],
  "send_rates": [150, 180, 210, 240, 270, 300, 330, 360, 390],
  "num_clients": 4,
  "num_queries": 100000,
  "num_trials": 3,
  "background_traffic": [4]
```
In addition to the general AWS setup, the following configuration parameters
will be run:

```
redundancy_mode=equal, send_rate=150,
redundancy_mode=coded, send_rate=150,
redundancy_mode=equal, send_rate=180,
redundancy_mode=coded, send_rate=180,
redundancy_mode=equal, send_rate=210,
redundancy_mode=coded, send_rate=210,
...
```

The loops that iterate over these combinations are found in [run_exp.py](../run_exp.py)
