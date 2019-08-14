**Note for AEC: In addition to the general description below, we provide
instructions for running experiments in [EXPERIMENTS.md](EXPERIMENTS.md)**

# ParM
ParM is a prediction serving system that realizes erasure-coding-based resilience by employing parity models. ParM is built off of [Clipper](https://github.com/ucbrise/clipper) (forked at version [0.3.0](https://github.com/ucbrise/clipper/commit/062b968f96c3821a354374f0b90d9c3776618419)).

## Requirements
In addition to [Clipper's](https://github.com/ucbrise/clipper) software requirements, ParM additionally introduces dependencies on:
* [OpenCV](https://github.com/opencv/opencv/tree/3.4.1) version 3.4.1

## Repository structure
Most of the core changes are made in [src](src). We make additional changes in [clipper_admin](clipper_admin) to ease deployment on distributed settings.

* [run](run): Contains scripts for running experiments with ParM. Please see the [documentation](run/README.md) therein for information on building appropriate Docker containers, accessing relevant AWS resoureces, and running experiments.

## Running ParM
Please see [EXPERIMENTS](EXPERIMENTS.md) for instructions on running ParM.
