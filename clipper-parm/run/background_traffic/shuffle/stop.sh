#!/bin/bash

NUMARG=2

if [ "$#" -ne "$NUMARG" ]; then
  echo "Usage: $0 <master_ip_file> <worker_ip_file>"
  exit 1
fi

master_ip_file=$1
worker_ip_file=$2

function join_by { echo "$*"; }

master_ip=$(cat $master_ip_file)
read -r -a ips < $worker_ip_file
ip_list=$(join_by "${ips[@]}")

for ip in $master_ip $ip_list; do
  ssh -n -o StrictHostKeyChecking=no ubuntu@$ip "\
    cd ~/clipper-parm/run/background_traffic/shuffle; \
    ./kill.sh" &
done
wait
