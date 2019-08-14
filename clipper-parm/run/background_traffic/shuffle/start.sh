#!/bin/bash

NUMARG=4

if [ "$#" -ne "$NUMARG" ]; then
  echo "Usage: $0 <master_ip_file> <worker_ip_file> <num_outstanding> <num_model_instances>"
  exit 1
fi

master_ip_file=$1
worker_ip_file=$2
num_outstanding=$3
num_model_instances=$4

function join_by { echo "$*"; }

master_ip=$(cat $master_ip_file)
read -r -a ips < $worker_ip_file
ip_list=$(join_by "${ips[@]}")

num_ips=${#ips[@]}
world_size=$((num_ips+1))

for ip in $ip_list; do
  ssh -n -o StrictHostKeyChecking=no ubuntu@$ip "\
    cd ~/clipper-parm/run/background_traffic/shuffle; \
    rm /home/ubuntu/bg_sock; \
    nohup python3 worker.py ${num_outstanding} --worker_ips ${ip_list} > ~/log_shuffling.txt 2>&1 &" &
done
wait

sleep 5
ssh -o StrictHostKeyChecking=no ubuntu@$master_ip "\
    cd ~/clipper-parm/run/background_traffic/shuffle; \
    rm /home/ubuntu/bg_sock; \
    nohup python3 master.py ${num_outstanding} ${num_model_instances} --worker_ips ${ip_list} > ~/log_shuffling.txt 2>&1 &"
