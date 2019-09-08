#!/bin/bash

# This script should be run from the `parity-models/train` directory.
# This script will download the base models used for training.

aws s3 cp s3://parity-models/base-models/ base_model_trained_files --recursive
