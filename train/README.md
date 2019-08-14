**Note for AEC: In addition to the general description below, we provide
instructions for running experiments in [EXPERIMENTS.md](EXPERIMENTS.md)**

# Training parity models
This repository contains the code used for training parity models.

## Background
The figure below shows an example of our target setup:
![alt text](img/parity_model_setup.png "Parity Model Setup")

Consider two copies of a machine learning model that have been deployed for inference
on separate servers. We call these models "base models." The overall goal of performing
inference is to perform inference over copies of the base model in response to queries
in order to return predictions.

We would like this setup to be resilient to one of the servers holding a copy of the
base model experiencing slowdown or failure. In order to do so, we will add a third
model, called a "parity model," along with an encoder and a decoder. The encoder
will operate over two queries dispatched to the base models and construct a "parity query."
The parity query will be dispatched for inference over the parity model in order
to return a parity prediction. The decoder uses the parity prediction along with
the any one out of two predictions that are to be returned from the base models
in order to reconstruct the unavailable prediction. In the example above, the
prediction from the second copy of the base model is slow/failed. The decoder
uses the prediction from the first copy of the base model along with the
parity prediction in order to reconstruct the second, unavailable, prediction.

This repository contains the framework for training a parity model in order
to enable accurate reconstruction of unavailable predictions.

### General parameters
More generally, we can have `k` copies of a deployed model and we would like to
be resilient to `r` of these copies being "unavailable" (i.e., slow/failed).
The encoder `E` takes as input `k` queries and returns `r` parity queries. These
`r` parity queries are dispatched to `r` different parity models. The decoder
takes as input any `k` out of the total `(k+r)` possible base model and parity
model predictions in order to reconstruct any `r` predictions being unavailable.
This project has mostly looked into the case where `r=1`, that is, when only
one prediction is unavailable at a time.

### Training a parity model
Consider the following simple setup. Let `F` denote the base model over which
we'd like to impart resilience and `Fp` denote a parity model. Suppose we
are parameterized with `k=3`, and denote `X1`, `X2`, and `X3` as queries,
with corresponding predictions being `F(X1)`, `F(X2)`, `F(X3)`. Let the
encoder perform summation as encoding: parity `P` is constructed as
`P = X1 + X2 + X3`. Let the decoder perform subtraction in attempt to
recover an unavailable prediction: if `F(X2)` is unavailable, the decoder
will attempt to reconstruct it as `F(X2)_recon = Fp(P) - F(X1) - F(X2)`.

From the encoder and decoder described above, we can see that accurate
reconstruction `F(X2)_recon = F(X2)` can be achieved when it is the case that:
`Fp(X1 + X2 + X3) = F(X1) + F(X2) + F(X3)`. In order to train a parity model `Fp`,
we can therefore sample many different combinations of `X1`, `X2`, and `X3` from
a target dataset, compute `Fp(X1 + X2 + X3)`, and compare it to ` F(X1) + F(X2) + F(X3)`
using some distance metric as a loss function.

## This repository
### Software requirements
This repository was developed using the following versions of software and has
not been tested using other versions.
* Python 3.6.5
* PyTorch version 1.0.0 and torchvision
* Other packages in `requirements.txt`

We suggest using our provided [Dockerfile](dockerfiles/ParityModelDockerfile),
which installs all necessary prerequisites. We have tested this using Docker version 19.03.1.
To build the image associated with this Dockerfile, run:
```bash
cd dockerfiles
docker build -t parity-models-pytorch -f ParityModelDockerfile .
```

If you would like to train on a GPU using the provided docker container, then
you will need to install [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker).

### Repository structure
* [base_models](base_models): Implementations of different 
* [base_model_trained_files](base_model_trained_files): PyTorch model state dictionaries containing
  trained parameters for the base models.
* [coders](coders): Implementations of simple encoding and decoding functions.
* [config](config): Configurations for training using the datasets shown in evaluation.
* [cub-localization](cub-localization): Instructions for training a parity
model for localization tasks.
* [data](data): Directory to which datasets will be downloaded.
* [datasets](datasets): PyTorch `Dataset` implementations for generating samples for training
  the encoding and decoding functions.
* [util](util): Utility methods used throughout the repository.
* [parity_model_trainer.py](parity_model_trainer.py): Top-level class for training a parity model.
* [train_config.py](train_config.py): Script to configure and launch a training run.

## Running a training job
Please see [EXPERIMENTS](EXPERIMENTS.md) for detail on running the training code.

## Making additions to this repository
Want to explore adding a new encoder, decoder, base model, dataset, etc.?
If so, check out the links below and raise an issue if you find the framework
poorly suited for your desired change!

### Adding a new base model
See details in the [base_models](base_models) directory regarding
adding a base model that is not included in this repository.

### Adding a new dataset
See the patterns used for constructing datasets in the [datasets](datasets) directory.

### Adding new encoders/decoders
See details in the [coders](coders) directory regarding adding a new
encoder/decoder architecture.
