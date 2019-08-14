**NOTE for AEC: Please see [EXPERIMENTS](EXPERIMENTS.md) for information on scripts used
for launching experiments.**

# Parity Models
This repository contains the code used for developing and evaluating parity
models. A parity model is a computational unit that enables the use of ideas
from erasure coding to impart resilience against slowdowns and failures that
occur in distributed data processing and serving environments.

This repository focuses particularly on using ideas from erasure codes to
impart resilience to prediction serving systems performing inference with
neural networks. For a brief overview of how a parity model enables this
resilience, please see this [description](train/README.md). We will add a
broader explanation of the function of a parity model to this README in the
near future.

## Repository structure
* [train](train): Code for training a neural network parity model
* [clipper-parm](clipper-parm): Code for ParM, a prediction serving system that
employs parity models to impart erasure-coding-based resilience to slowdowns
and failures.

## License
```
Copyright 2019, Carnegie Mellon University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
