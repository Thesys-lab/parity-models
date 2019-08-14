#!/bin/bash

NUMARG=16
if [ "$#" -ne "$NUMARG" ]; then
  echo "Usage: $0 <s3_save_path> <redundancy_mode> <queue_mode> <num_models> <num_red> <ec_k_val> <batch_size> <model> <num_groups_per_client> <num_outstanding> <rate> <img_dir> <num_imgs> <mode> <do_background> <num_ignore>"
  exit 1
fi

echo $@

s3_save_path=$1
red_mode=$2
queue_mode=$3
num_models=$4
num_red=$5
ec_k_val=$6
batch_size=$7
model=$8
num_groups_per_client=${9}
num_outstanding=${10}
rate=${11}
mode=${14}
do_background=${15}
num_ignore=${16}

num_per_group=$ec_k_val

port=7001
client_port=8000

sudo $PARM_LOCAL_HOME/util/stop_redis_server.sh

echo "Enabling docker forwarding"
#sudo iptables -P FORWARD ACCEPT

cd $PARM_LOCAL_HOME/clipper_admin
sudo python3 setup.py install &> /dev/null
cd -
CURDIR=$(pwd)

echo "Stopping clipper and cleaning"
$PARM_LOCAL_HOME/util/kill_frontend.sh
python3 $PARM_LOCAL_HOME/util/stop_clipper.py
$PARM_LOCAL_HOME/util/clean-docker.sh &> /dev/null

script_name=local_exp.py

cd ${PARM_LOCAL_HOME}/dockerfiles/parm-dockerfiles
./build_pytorch_images.sh &> ${PARM_LOCAL_HOME}/run/local_exp/dockerbuild.txt
${PARM_LOCAL_HOME}/util/kill_worker.sh &> /dev/null
../util/clean-docker.sh &> /dev/null

total_num_workers=$((num_models+num_red))
worker_ip_file=${PARM_LOCAL_HOME}/run/local_exp/worker_ips.txt
worker_ports_file=${PARM_LOCAL_HOME}/run/local_exp/worker_ports.txt
echo "" > $worker_ip_file
echo "" > $worker_ports_file
echo "Starting workers"
for ((i = 0; i < $total_num_workers; i++)); do
  if [ "$i" -ge "$num_models" ]; then
    redundant=--w_redundant
    echo "Starting redundant worker on ${ips[i]}"
  else
    redundant=''
    echo "Starting worker on ${ips[i]}"
  fi

  worker_port=$(($port + $i))
  worker_log=${PARM_LOCAL_HOME}/run/local_exp/log_worker${i}.txt

  echo "localhost" >> $worker_ip_file
  echo "${worker_port}" >> $worker_ports_file
  ssh -n localhost "\
    echo 'Starting ${i}'; \
    cd ${PARM_LOCAL_HOME}/run; \
    nohup python3 ${script_name} worker ${red_mode} ${queue_mode} ${model} \
                            --batch_size ${batch_size} \
                            --w_port ${worker_port} \
                            --w_qf_hostname localhost ${redundant} > ${worker_log} 2>&1 &" &
done
wait

client_ip_file=${PARM_LOCAL_HOME}/run/local_exp/client_ips.txt
client_ports_file=${PARM_LOCAL_HOME}/run/local_exp/client_ports.txt
echo "" > $client_ip_file
echo "" > $client_ports_file
img_dir=${PARM_LOCAL_HOME}/images/local
num_imgs=1
cid=0
for cip in localhost; do
  echo $cip
  this_client_port=$(($client_port + $cid))

  echo "${cip}" >> $client_ip_file
  echo "${this_client_port}" >> $client_ports_file

  ssh -n localhost "\
    cd ${PARM_LOCAL_HOME}/run; \
    nohup python3 client.py --num_groups ${num_groups_per_client} \
                            --num_groups_outstanding ${num_outstanding} \
                            --num_batches_per_group ${num_per_group} \
                            --batch_size ${batch_size} \
                            --frontend_ip localhost \
                            --port ${this_client_port} \
                            --img_dir ${img_dir} \
                            --num_imgs ${num_imgs} \
                            --rate ${rate} \
                            --frontend_private_ip localhost > ${PARM_LOCAL_HOME}/run/local_exp/log_client${cid}.txt 2>&1 &" &
  cid=$(($cid + 1))
done
wait

sleep_time=40
echo "Sleeping for ${sleep_time} seconds"
sleep $sleep_time

echo "Starting frontend"

out_path=/tmp/${s3_save_path}
stats_path=${out_path}/stats_files
mkdir -p ${stats_path}

f_ips=$(cat $worker_ip_file)
f_ports=$(cat $worker_ports_file)
c_ips=$(cat $client_ip_file)
c_ports=$(cat $client_ports_file)
cd ${PARM_LOCAL_HOME}/run
python3 ${script_name} \
  frontend $red_mode $queue_mode $model --num_models $num_models \
  --num_redundant_models $num_red --batch_size $batch_size \
  --f_ips $f_ips --f_ports $f_ports \
  --f_client_ips $c_ips --f_client_ports $c_ports \
  --f_mode $mode --f_outfile $stats_path/total_time.txt | tee ${PARM_LOCAL_HOME}/run/local_exp/frontend_log.txt

${PARM_LOCAL_HOME}/util/kill_worker.sh

echo "Getting frontend log"
log_file=$out_path/log.txt
meta_file=$out_path/meta.txt
num_clients=${#client_ips[@]}
echo "NUM CLIENTS IS ${num_clients}"
docker logs query_frontend-parm &> $log_file
echo "red_mode=${red_mode} queue_mode=${queue_mode} num_models=${num_models} batch_size=${batch_size} model=${model} num_clients=${num_clients} num_groups_per_client=${num_groups_per_client} num_outstanding=${num_outstanding} ec_k_val=${ec_k_val} num_ignore=${num_ignore}" > $meta_file
cd ${PARM_LOCAL_HOME}/stats/
echo "${log_file} ${stats_path} ${red_mode} ${num_clients} ${ec_k_val} ${num_ignore}"

cd $out_path
tmpout=/tmp/${s3_save_path}
mkdir -p $tmpout
tar -czf $tmpout/exp_files.tar.gz *
cd -
echo "${tmpout}/exp_files.tar.gz"
sleep 30
