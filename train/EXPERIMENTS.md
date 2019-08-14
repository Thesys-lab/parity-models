# Training experiments
We first describe how to run a simple training experiment and then provide
details for peforming all training runs found in evaluation.

## Setup
We suggest using our provided [Dockerfile](./dockerfiles/ParityModelDockerfile),
which installs all necessary prerequisites. We have tested this using Docker version 19.03.1.
To build the image associated with this Dockerfile, run:
```bash
cd dockerfiles
docker build -t parity-models-pytorch -f ParityModelDockerfile .
```
If you would like to train on a GPU using the provided docker container, then
you will need to install [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker).

## Quickstart
We first show how to run an example and then provide
instructions for other datasets and models.

### MNIST Example
This example trains an MLP parity model for 10 epochs using the MNIST dataset.
This should take less than 5 minutes on a laptop.

1. Ensure that you have built the `parity-models-pytorch` Docker image.
2. Start the Docker container, and attach the `parity-models` volume. **NOTE:** If you want your container to have access to a GPU, then you must substitute the `docker` command below for `nvidia-docker`.
```bash
docker run -it --rm -v /path/to/parity-models:/workspace/parity-models parity-models-pytorch:latest
```
3. Start training the MNIST parity model:
```bash
# These commands should be run from within the Docker container
cd /workspace/parity-models/train
python3 train_config.py config/example.json save
```

**What output should you see?**
Provided all has gone well, you should see the following output appear:
```
mnist base-mlp 2 mse add sub
Base model train accuracy is 59754 / 60000 = 0.9959
Base model val accuracy is 59754 / 60000 = 0.9959
Base model test accuracy is 9793 / 10000 = 0.9793
Epoch 0. train. Top-1=0.1634, Top-5=0.6181, Loss=163.7766: 100%|###############################################################################| 430/430 [00:02<00:00, 196.08it/s]
Epoch 0. val. Top-1=0.2862, Top-5=0.7554, Loss=116.5788: 100%|###################################################################################| 40/40 [00:00<00:00, 315.57it/s]
Epoch 1. train. Top-1=0.4334, Top-5=0.8611, Loss=82.6607: 100%|################################################################################| 430/430 [00:02<00:00, 183.46it/s]
Epoch 1. val. Top-1=0.5138, Top-5=0.9122, Loss=66.1130: 100%|####################################################################################| 40/40 [00:00<00:00, 315.98it/s]
Epoch 2. train. Top-1=0.5585, Top-5=0.9221, Loss=62.2081:  36%|############################6                                                   | 154/430 [00:00<00:01, 195.82it/s
```
Let's walk through these step-by-step:
* `mnist base-mlp 2 mse add sub`: This line prints information about the
   configuration currently being trained. In this case, this indicates that
   we are training with the MNIST dataset using an MLP model, with k=2,
   using mean-squared-error as loss, using the addition encoder and subtraction
   decoder.
* `Base model train accuracy is 59754 / 60000 = 0.9959`: These lines show the
   accuracy of the base model (`F` in [Background](#Background)) on the
   training, validation, and test datasets. Note that the accuracy for the
   training and validation sets will always look the same -- we partition the
   official training datasets into a "training" and "validation" split after
   this line has been printed. Sorry for this being confusing :(
* `Epoch 0. train. ...`: When these lines run, we're actually training! For
   each epoch, we print the current progress, top-1 and top-5 accuracies,
   and current average loss. We print these for both training and validation
   sets. Accuracy on the test dataset is calculated, but not printed to the
   user.

At the end of each epoch, the current results (e.g., loss, accuracy) are saved
to files under the `save` directory (or whichever directory you passed in to
the training command above). Results will be placed in a directory indexed
by the configuration of our run. In the example above, this will be in:
```
save/mnist/base-mlp/k2/mse/add/sub/
```

This directory contains a number of files:
* `current.pth`: Saved model and optimizer parameters from the last epoch to
complete
* `best.pth`: Saved model and optimizer parameters from the epoch which
resulted in the highest accuracy on the validation dataset.
* `{train,val,test}_overall_{top1,top2,top5,top10,top20}`: The accuracy on the
`{train,val,test}` datasets obtained by comparing the result of the decoder to
the true label from the dataset. Each line represents the result from an epoch.
* `{train,val,test}_reconstruction_{top1,top2,top5,top10,top20}`: The accuracy on the
`{train,val,test}` datasets obtained by comparing the result of the decoder to
the result that would be returned by the base model (`F`).
 Each line represents the result from an epoch.

The files in this directory enable one to continue training from a checkpoint.
Using the example above, we can continue training from the last epoch of our
previous run with:
```bash
python3 train_config.py save --continue_from_file save/mnist/base-mlp/k2/mse/add/sub/current.pth
```

### Using other datasets and base models
Due to the size of the datasets and model files used in our evaluation, we are
unable to keep all datasets and models on Github. We have hosted the remaining
datasets and model weights in a public AWS S3 bucket. Please follow the
commands below to access these.

**Download the datasets used in our evaluation:**
```bash
./download_data.sh
```
This will download the Google Commands and Cat v. Dog datasets. Remaining
datasets are handled by PyTorch and downloaded as necessary.

**Download trained base models**

Download the trained base model weights used in training a parity model. Some
of these weights are already located in [base_model_trained_files](base_model_trained_files).
Others (those too large to be checked-in with Github) may be downloaded with:
```
./download_base_models.sh
```

Once these datasets and base models have been downloaded, results found in the
paper may be reproduced by running:
```bash
./experiments.sh
```
The output and structure of training should look similar to that in Quickstart
above.

This will save all accuracies and trained model files under the `save` directory
(the script will create this).

**NOTE:** This script will run many training configurations and many epochs and
will likely take days to complete. If you are primarily interested in running a
simple example, please see our [MNIST example](README.md).
