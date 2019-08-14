#!/bin/bash

# This script sets up a barebones Ubuntu 16.04 machine to contain all software
# needed to execute ParM locally.

if [ "$#" -eq 0 ]; then

  cd ~
  sudo apt-get update
  sudo apt-get install -y automake autoconf autoconf-archive libtool libboost-all-dev \
    libevent-dev libdouble-conversion-dev libgoogle-glog-dev libgflags-dev liblz4-dev \
    liblzma-dev libsnappy-dev make zlib1g-dev binutils-dev libjemalloc-dev libssl-dev \
    pkg-config libiberty-dev git cmake libev-dev libhiredis-dev libzmq5 libzmq5-dev build-essential
  sudo apt-get install -y python3-pip
  pip3 install numpy docker kubernetes torch==1.0.0 torchvision==0.2.1 cloudpickle==0.5.*

  # Install Folly
  git clone https://github.com/facebook/folly
  cd folly/folly
  git checkout tags/v2017.08.14.00
  autoreconf -ivf
  ./configure
  make -j4
  sudo make install
  cd ../..

  # Install Cityhash
  git clone https://github.com/google/cityhash
  cd cityhash
  ./configure
  make all check CXXFLAGS="-g -O3"
  sudo make install
  cd ..

  # Install OpenCV
  cd ~
  git clone https://github.com/opencv/opencv.git
  cd opencv
  git checkout 3.4.1
  mkdir build
  cd build
  cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
  make -j7
  sudo make install
  cd ~

  # Install docker
  sudo apt-get install -y \
      apt-transport-https \
      ca-certificates \
      curl \
      gnupg-agent \
      software-properties-common

  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
  sudo apt-key fingerprint 0EBFCD88
  sudo add-apt-repository \
     "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
     $(lsb_release -cs) \
     stable"
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io
  sudo groupadd docker
  sudo usermod -aG docker $USER

  echo "======= Docker install complete ======="
  echo "Please log out and back for non-root docker permissions to take effect"
  echo "You may then continue setup by running:"
  echo "    ./setup.sh continue"
else
  echo "Continuing setup"
  ln -s ~/parity-models/clipper-parm /home/ubuntu/clipper-parm
  cd ~/parity-models/clipper-parm/dockerfiles
  ./build-parm-lib-base.sh
  cd parm-dockerfiles
  ./build_frontend_image.sh
  ./build_pytorch_images.sh
  cd ~
  ssh-keygen -f ~/.ssh/id_rsa -t rsa -N ''
  cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
  ssh localhost
fi
