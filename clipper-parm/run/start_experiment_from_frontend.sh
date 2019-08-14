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
img_dir=${12}
num_imgs=${13}
mode=${14}
do_background=${15}
num_ignore=${16}

num_per_group=$ec_k_val

worker_ip_file=~/workers.txt
private_frontend_ip_file=~/private_frontend_ip.txt
private_worker_ip_file=~/private_worker_ip.txt

ssh -o StrictHostKeyChecking=no ubuntu@$(cat $private_frontend_ip_file) "echo 'hi'"

function join_by { echo "$*"; }
qf_ip=$(cat ~/frontend.txt)
qf_private_ip=$(cat $private_frontend_ip_file)
read -r -a ips < ~/workers.txt
read -r -a client_ips < ~/clients.txt
port=7000
client_port=7000


PARM_REMOTE_HOME=/home/ubuntu/clipper-parm

echo "Stopping redis-server if running"
sudo $PARM_REMOTE_HOME/util/stop_redis_server.sh

echo "Enabling docker forwarding"
sudo iptables -P FORWARD ACCEPT

cd $PARM_REMOTE_HOME/clipper_admin
sudo python3 setup.py install &> /dev/null
cd -

echo "Stopping clipper and cleaning"
$PARM_REMOTE_HOME/util/kill_frontend.sh
python3 $PARM_REMOTE_HOME/util/stop_clipper.py
$PARM_REMOTE_HOME/util/clean-docker.sh &> /dev/null

ip_list=$(join_by "${ips[@]}")
client_ip_list=$(cat ~/clients.txt)
port_list=$port
total_num_workers=$((num_models+num_red))
echo "NUM WORKERS IS ${total_num_workers}"
if [ "${do_background}" -gt "0" ]; then
  cd background_traffic/shuffle/
  echo "Starting background shuffles"
  #./stop.sh $private_frontend_ip_file $private_worker_ip_file
  ./start.sh $private_frontend_ip_file $private_worker_ip_file $do_background $total_num_workers
  cd -
fi

script_name=pytorch_distributed.py

echo "Starting workers"
for ((i = 0; i < $total_num_workers; i++)); do
  if [ "$i" -ge "$num_models" ]; then
    redundant=--w_redundant
    echo "Starting redundant worker on ${ips[i]}"
  else
    redundant=''
    echo "Starting worker on ${ips[i]}"
  fi

  scp -o StrictHostKeyChecking=no ${script_name} ubuntu@${ips[i]}:$PARM_REMOTE_HOME/run/
  ssh -n -o StrictHostKeyChecking=no ubuntu@${ips[i]} "\
    echo 'Starting ${i}'; \
    cd ${PARM_REMOTE_HOME}/dockerfiles/parm-dockerfiles; \
    ./build_pytorch_images.sh &> /home/ubuntu/dockerbuild.txt; \
    ${PARM_REMOTE_HOME}/util/kill_worker.sh &> /dev/null; \
    cd ${PARM_REMOTE_HOME}/clipper_admin; \
    sudo python3 setup.py install &> /dev/null; \
    ../util/clean-docker.sh &> /dev/null; \
    cd ../run; \
    nohup python3 ${script_name} worker ${red_mode} ${queue_mode} ${model} \
                            --batch_size ${batch_size} \
                            --w_port ${port} \
                            --w_qf_hostname ${qf_ip} ${redundant} > ~/log_worker.txt 2>&1 &" &
done
wait

echo "Starting clients at $client_ip_list"
for cip in ${client_ip_list}; do
  echo $cip
  scp -o StrictHostKeyChecking=no client.py ubuntu@$cip:$PARM_REMOTE_HOME/run/
  ssh -n -o StrictHostKeyChecking=no ubuntu@$cip "\
    ${PARM_REMOTE_HOME}/util/kill_worker.sh; \
    cd ${PARM_REMOTE_HOME}/clipper_admin; \
    sudo python3 setup.py install &> /dev/null; \
    cd ../run; \
    nohup python3 client.py --num_groups ${num_groups_per_client} \
                            --num_groups_outstanding ${num_outstanding} \
                            --num_batches_per_group ${num_per_group} \
                            --batch_size ${batch_size} \
                            --frontend_ip ${qf_ip} \
                            --port ${client_port} \
                            --img_dir ${img_dir} \
                            --num_imgs ${num_imgs} \
                            --rate ${rate} \
                            --frontend_private_ip ${qf_private_ip} > ~/log_client.txt 2>&1 &" &
done
wait

sleep_time=40
echo "Sleeping for ${sleep_time} seconds"
sleep $sleep_time

echo "Starting frontend"

out_path=/tmp/${s3_save_path}
stats_path=${out_path}/stats_files
mkdir -p ${stats_path}

echo "CLIENT IP LIST: ${client_ip_list}"

# If redundancy mode is "none", then the frontend should only use the first
# `num_models` ip addresses for workers.
if [ "$red_mode" == "none" ]; then
  f_ip_list=$(echo ${ips[@]:0:$num_models})
else
  f_ip_list=$ip_list
fi

python3 ${script_name} \
  frontend $red_mode $queue_mode $model --num_models $num_models \
  --num_redundant_models $num_red --batch_size $batch_size \
  --f_ips $f_ip_list --f_ports $port_list \
  --f_client_ips $client_ip_list --f_client_ports $client_port \
  --f_mode $mode --f_outfile $stats_path/total_time.txt | tee ~/frontend_log.txt

echo "Killing workers"
for ip in $ip_list; do
  echo "Killing at ${ip}"
  ssh -o StrictHostKeyChecking=no ubuntu@$ip "${PARM_REMOTE_HOME}/util/kill_worker.sh"
done

echo "Killing clients"
client_path=${out_path}/clients
mkdir $client_path
client_id=0
for ip in $client_ip_list; do
  echo "Killing at ${ip}"
  ssh -o StrictHostKeyChecking=no ubuntu@$ip "${PARM_REMOTE_HOME}/util/kill_worker.sh"
  scp -o StrictHostKeyChecking=no ubuntu@$ip:~/log_client.txt $client_path/log_client${client_id}.txt
  scp -o StrictHostKeyChecking=no ubuntu@$ip:~/sleep.txt $client_path/sleep${client_id}.txt
  client_id=$((client_id+1))
done

if [ "${do_background}" -gt "0" ]; then
  cd background_traffic/shuffle/
  ./stop.sh $private_frontend_ip_file $private_worker_ip_file
  cd -
fi

echo "Getting frontend log"
log_file=$out_path/log.txt
meta_file=$out_path/meta.txt
num_clients=${#client_ips[@]}
echo "NUM CLIENTS IS ${num_clients}"
docker logs query_frontend-parm &> $log_file
echo "red_mode=${red_mode} queue_mode=${queue_mode} num_models=${num_models} batch_size=${batch_size} model=${model} num_clients=${num_clients} num_groups_per_client=${num_groups_per_client} num_outstanding=${num_outstanding} ec_k_val=${ec_k_val} num_ignore=${num_ignore}" > $meta_file
echo "${log_file} ${stats_path} ${red_mode} ${num_clients} ${ec_k_val} ${num_ignore}"

cd $out_path
tmpout=/tmp/${s3_save_path}
mkdir -p $tmpout
tar -czf $tmpout/exp_files.tar.gz *
cd -
echo "${tmpout}/exp_files.tar.gz"
sleep 30
