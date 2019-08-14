#!/bin/bash
exp_id=sosp19-round1
final_dir=bg
frontend_type=c5.xlarge
worker_type=p2.xlarge
client_type=m5.large
model=resnet18
num_workers=18
num_clients=1
num_models=2
batch_size=1
total_send_rate=28
build_mode=release
reuse=1
img_dir=/home/ubuntu/cat_v_dog/test1
num_img=500
red_mode=equal
queue_mode=single_queue
rate=28

worker_ip_file=~/workers.txt
private_frontend_ip_file=~/private_frontend_ip.txt

PARM_REMOTE_HOME=/home/ubuntu/clipper-parm

ssh -o StrictHostKeyChecking=no ubuntu@$(cat $private_frontend_ip_file) "echo 'hi'"

function join_by { echo "$*"; }
qf_ip=$(cat ~/frontend.txt)
qf_private_ip=$(cat $private_frontend_ip_file)
read -r -a ips < ~/workers.txt
read -r -a client_ips < ~/clients.txt
port=7000
client_port=7000

ip_list=$(join_by "${ips[@]}")
client_ip_list=$(cat ~/clients.txt)
read -r -a private_ips < ~/private_worker_ip.txt
private_ip_list=$(join_by "${private_ips[@]}")
f_ip_list=$private_ip_list

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
$PARM_REMOTE_HOME/util/clean-docker-bg.sh &> /dev/null

script_name=background.py

echo "Starting workers"
num_launch=$num_workers
if [ "$num_models" -gt "$num_workers" ]; then
  num_launch=$num_models
fi
echo "Num launch is ${num_launch}"
for ((i = 0; i < $num_launch; i++)); do
  ip_idx=$(( $i % $num_workers ))
  echo "Starting worker on ${ips[ip_idx]}, idx=${i}, ip_idx=${ip_idx}"
  scp -o StrictHostKeyChecking=no ${script_name} ubuntu@${ips[ip_idx]}:$PARM_REMOTE_HOME/run/
  ssh -n -o StrictHostKeyChecking=no ubuntu@${ips[ip_idx]} "\
    echo 'Starting ${i}'; \
    ${PARM_REMOTE_HOME}/util/kill_worker.sh &> /dev/null; \
    cd ${PARM_REMOTE_HOME}/clipper_admin; \
    sudo python3 setup.py install &> /dev/null; \
    ../util/clean-docker-bg.sh &> /dev/null; \
    cd ../run; \
    nohup python3 ${script_name} worker ${red_mode} ${queue_mode} ${model} \
                                 --batch_size ${batch_size} --num_models ${num_models} --num_workers ${num_workers} \
                                 --w_idx ${i} --w_qf_hostname ${qf_ip} > ~/log_bg_worker.txt 2>&1 &" &
done
wait

echo "Starting clients at $client_ip_list"
for cip in ${client_ip_list}; do
  echo $cip
  scp -o StrictHostKeyChecking=no background_client.py ubuntu@$cip:$PARM_REMOTE_HOME/run/
  ssh -n -o StrictHostKeyChecking=no ubuntu@$cip "\
    ${PARM_REMOTE_HOME}/util/kill_worker.sh; \
    cd ${PARM_REMOTE_HOME}/clipper_admin; \
    sudo python3 setup.py install &> /dev/null; \
    cd ../run; \
    nohup python3 background_client.py \
                            --frontend_ip ${qf_ip} --port ${client_port} \
                            --img_dir ${img_dir} --num_imgs ${num_img} \
                            --rate ${rate} > ~/log_client.txt 2>&1 &" &
done
wait

sleep_time=60
echo "Sleeping for ${sleep_time} seconds"
sleep $sleep_time

echo "Starting frontend"

out_path=/tmp/${s3_save_path}
stats_path=${out_path}/stats_files
mkdir -p ${stats_path}

echo "CLIENT IP LIST: ${client_ip_list}"

# Remove any existing file that conflicts with the unix domain socket
rm /home/ubuntu/bg_sock

# Start the background frontend. This will run until stop_background.py is
# executed on the frontend.
python3 ${script_name} \
  frontend $red_mode $queue_mode $model --num_models $num_models \
  --batch_size $batch_size --f_ips $f_ip_list \
  --f_client_ips $client_ip_list --f_client_ports $client_port \
  --f_mode $build_mode | tee ~/frontend_log.txt

echo "Killing workers"
for ip in $ip_list; do
  echo "Killing at ${ip}"
  ssh -o StrictHostKeyChecking=no ubuntu@$ip "${PARM_REMOTE_HOME}/util/kill_worker.sh"
done

echo "Killing clients"
for ip in $client_ip_list; do
  echo "Killing at ${ip}"
  ssh -o StrictHostKeyChecking=no ubuntu@$ip "${PARM_REMOTE_HOME}/util/kill_worker.sh"
done
