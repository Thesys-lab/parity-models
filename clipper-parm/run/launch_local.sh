#!/bin/bash

NUMARG=9
if [ "$#" -ne "$NUMARG" ]; then
  echo "Usage: $0 <red_mode> <queue_mode> <ec_k_val> <num_models>"
  echo "          <num_red_models> <num_clients> <build_mode> <final_dir>"
  echo "          <config_file>"
  echo "                    If reusing, must set reuse=1"
  echo "  red_mode:         One of {none, equal, coded, cheap}"
  echo "  queue_mode:       One of {single_queue, rr}"
  echo "  ec_k_val:         K value to use in coding"
  echo "  num_models:       Number of worker instances to launch (both parity and not)"
  echo "  num_red_models:   Number of redundant worker instances to launch."
  echo "  num_clients:      Number of clients to launch"
  echo "  build_mode:       One of {debug, release}"
  echo "  final_dir:        Final directory to save to"
  echo "  config_file:      File containing json configuration"
  exit 1
fi

red_mode=$1
queue_mode=$2
ec_k_val=$3
num_models=$4
num_red_models=$5
num_clients=${6}
build_mode=${7}
final_dir=${8}
config_file=${9}
my_final_dir=$final_dir/0

if [ "$build_mode" != "debug" ] && [ "$build_mode" != "release" ]; then
  echo "Unrecognized build mode ${build_mode}"
  exit 1
fi

mkdir local_exp

root_save_dir=/tmp/$final_dir
mkdir -p $root_save_dir

copy_dir=$root_save_dir/0
mkdir $copy_dir
my_dir=$copy_dir
clipper_idx=0

total_num_models=$num_models
echo "total_num_models is ${total_num_models}"

config_flag=--release
if [ "$build_mode" != "release" ]; then
  config_flag=''
fi

CURDIR=$(pwd)
cd ..
if [ ! -d ${build_mode} ]; then ./configure ${config_flag}; fi;
cd ${build_mode};
make -j$(nproc)

cd $CURDIR
python3 run_exp.py $config_file run --final_dir ${my_final_dir} --idx 0 --local
