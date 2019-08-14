#!/bin/bash

NUMARG=2
if [ "$#" -ne "$NUMARG" ]; then
  echo "Usage: $0 <remote_path> <local>"
  exit 1
fi

remote_path=$1
is_local=$2

if [ "$is_local" -eq 0 ]; then
  exp_dir=cur_exp
else
  exp_dir=local_exp
fi

results_dir=$(pwd)/$exp_dir/results
if [ ! -d $results_dir ]; then
  mkdir -p $results_dir
fi

local_dir=$results_dir/$remote_path
mkdir -p $local_dir

if [ "$is_local" -eq 0 ]; then
  scp -i $PARM_SSH_KEY ubuntu@$(cat cur_exp/frontend_ip.txt):/tmp/${remote_path}/exp_files.tar.gz $local_dir
else
  cp /tmp/${remote_path}/exp_files.tar.gz $local_dir
fi

curdir=$(pwd)
cd ../stats
./print_stats.sh $local_dir/exp_files.tar.gz
cd $curdir
