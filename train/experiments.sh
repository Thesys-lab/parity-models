#!/bin/bash

python3 train_config.py config/mnist.json save/
python3 train_config.py config/fashion-mnist.json save/
python3 train_config.py config/cifar10.json save/
python3 train_config.py config/cifar100.json save/
python3 train_config.py config/cat_v_dog.json save/
python3 train_config.py config/gcommands.json save/

cd cub-localization
python3 coded.py
cd ..
mv cub-localization/coded-k2 save/cub-localization
