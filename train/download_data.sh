#!/bin/bash

# This script should be run from the `parity-models/train` directory.
# This script will download the Cat v. Dog dataset that we use as well as the
# Google Commands dataset.

# We use the PyTorch dataloading classes to download CIFAR-100, CIFAR-10,
# Fashion-MNIST, and MNIST. At runtime, PyTorch will detect whether these
# datasets have been downloaded, and download them if necessary.

# Download Cat v. Dog dataset
cat_dog_dir=data/cat_v_dog
if [ ! -d $cat_dog_dir ]; then
  aws s3 cp s3://parity-models/datasets/cat_v_dog.zip .
  unzip -q cat_v_dog.zip
  mv cat_v_dog $cat_dog_dir
  rm cat_v_dog.zip
fi

cub_dir=cub-localization/src/data/images
if [ ! -d $cub_dir ]; then
  aws s3 cp s3://parity-models/datasets/cub-localization.zip .
  unzip -q cub-localization.zip
  mv cub-localization/images cub-localization/src/data/
  rm cub-localization.zip
fi

# Download Google Commands dataset
gcommands_dir=data/gcommands
if [ ! -d $gcommands_dir ]; then
  tmpgcommand=tmpgcommand
  mkdir $tmpgcommand
  gcommand_tar=speech_commands_v0.01.tar.gz
  wget http://download.tensorflow.org/data/$gcommand_tar
  mv $gcommand_tar $tmpgcommand
  cd $tmpgcommand
  tar -xf $gcommand_tar
  cd ..

  mkdir -p $gcommands_dir
  python3 util/make_gcommands_dataset.py $tmpgcommand $gcommands_dir
  rm -rf $tmpgcommand
fi
