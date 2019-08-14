#!/bin/bash

NUMARG=10
if [ "$#" -ne "$NUMARG" ]; then
  echo "Usage: $0 <exp_id> <queue_mode> <frontend_type> <worker_type> <client_type> <num_workers> <num_clients> <reuse> <my_dir> <args_file>"
  echo "  exp_id:           Identifier of this experiment in tagging ec2 resources."
  echo "                    If reusing, must set reuse=1"
  echo "  queue_mode:       One of {single_queue, rr}"
  echo "  frontend_type:    Type of instance to use for frontend"
  echo "  worker_type:      Type of instance to use for frontend"
  echo "  client_type:      Type of instance to use for client"
  echo "  num_workers:      Number of workers to launch"
  echo "  num_clients:      Number of clients to launch"
  echo "  reuse:            Whether to reuse existing exp_id"
  echo "  my_dir:           Directory to save files to"
  echo "  args_file:        File containing arguments"
  exit 1
fi

exp_id=${1}
queue_mode=${2}
frontend_type=${3}
worker_type=${4}
client_type=${5}
num_workers=${6}
num_clients=${7}
reuse=${8}
my_dir=${9}
args_file=${10}

num_red_models=0
red_mode=equal
build_mode=release

reuse_flag=--reuse
if [ "$reuse" -eq "0" ]; then
  reuse_flag=''
fi

frontend_ip_file=$my_dir/bg_frontend_ips.txt
worker_ip_file=$my_dir/worker_ips.txt
client_ip_file=$my_dir/bg_client_ips.txt

all_worker_ip_file=$my_dir/worker_ips_all.txt
private_frontend_ip_file=$my_dir/bg_private-master_ips.txt
private_worker_ip_file=$my_dir/private-worker_ips.txt

# Create frontend instances
python3 launch_ec2.py \
  ${exp_id} bg_frontend 1 ${red_mode} \
  --instance_type ${frontend_type} \
  --save_ip_file ${frontend_ip_file} \
  --save_private_ip_file ${private_frontend_ip_file} \
  --save_instance_ids_file cur_exp/bg_frontend_ids.txt \
  ${reuse_flag}

# Create worker instances
python3 launch_ec2.py \
  ${exp_id} worker ${num_workers} ${red_mode} \
  --red_models ${num_red_models} \
  --instance_type ${worker_type} \
  --save_ip_file ${worker_ip_file} \
  --save_private_ip_file ${private_worker_ip_file} \
  --save_instance_ids_file cur_exp/bg_worker_ids.txt \
  ${reuse_flag}

# Create client instances
python3 launch_ec2.py \
  ${exp_id} bg_client ${num_clients} ${red_mode} \
  --instance_type ${client_type} \
  --save_ip_file ${client_ip_file} \
  --save_instance_ids_file cur_exp/bg_client_ids.txt \
  ${reuse_flag}

sleep 30

cp $frontend_ip_file cur_exp/bg_frontend_ip.txt

frontend_ip=$(cat $frontend_ip_file)
worker_ips=$(cat $worker_ip_file)
client_ips=$(cat $client_ip_file)

# This represents all workers and background workers
all_worker_ips=$(cat $all_worker_ip_file)

key=$PARM_SSH_KEY
ssh_opts="-o StrictHostKeyChecking=no -i ${key}"

# Generate ssh key on frontend and copy it locally
ssh $ssh_opts ubuntu@$frontend_ip "if [ ! -f ~/.ssh/id_rsa.pub ]; then ssh-keygen -f ~/.ssh/id_rsa -t rsa -N ''; fi"
ssh $ssh_opts ubuntu@$frontend_ip 'ssh-keygen -f "/home/ubuntu/.ssh/known_hosts" -R 0.0.0.0'
key_name=$my_dir/id_rsa_bg.pub
scp $ssh_opts ubuntu@$frontend_ip:~/.ssh/id_rsa.pub $key_name

for ip in $client_ips $all_worker_ips $frontend_ip; do
  scp $ssh_opts $key_name ubuntu@$ip:~/id_rsa_bg.pub &
done
wait

for wip in $all_worker_ips $client_ips $frontend_ip; do
  ssh $ssh_opts ubuntu@$wip "cat id_rsa_bg.pub >> ~/.ssh/authorized_keys" &
done
wait

PARM_REMOTE_HOME=/home/ubuntu/clipper-parm

# Rsync clipper source to frontend
config_flag=--release
if [ "$build_mode" != "release" ]; then
  config_flag=''
fi

rsync -zh -r -t -e "ssh -i ${key}" $PARM_LOCAL_HOME/src ubuntu@$frontend_ip:$PARM_REMOTE_HOME/
ssh $ssh_opts ubuntu@$frontend_ip "\
  cd ${PARM_REMOTE_HOME}; \
  if [ ! -d ${build_mode} ]; then ./configure ${config_flag}; fi; \
  cd ${build_mode}; \
  make -j4"

# Rsync clipper_admin to frontend and all workers, along with background traffic
echo "All worker ips are: ${all_worker_ips}"
for ip in $frontend_ip $worker_ips $(cat $client_ip_file); do
  echo $ip
  rsync -zh -r -e "ssh -i ${key}" $PARM_LOCAL_HOME/util ubuntu@$ip:$PARM_REMOTE_HOME/ &
  rsync -zh -r -e "ssh -i ${key}" $PARM_LOCAL_HOME/clipper_admin ubuntu@$ip:$PARM_REMOTE_HOME/ &
  rsync -zh -r -e "ssh -i ${key}" $PARM_LOCAL_HOME/dockerfiles ubuntu@$ip:$PARM_REMOTE_HOME/ &
  rsync -zh -r -e "ssh -i ${key}" $PARM_LOCAL_HOME/monitoring ubuntu@$ip:$PARM_REMOTE_HOME/ &
  rsync -zh -r -e "ssh -i ${key}" $PARM_LOCAL_HOME/containers/python ubuntu@$ip:$PARM_REMOTE_HOME/containers/ &
  rsync -zh -r -e "ssh -i ${key}" background_traffic ubuntu@$ip:$PARM_REMOTE_HOME/run/ &
done
wait

# Copy lists of ips to frontend
scp $ssh_opts $worker_ip_file ubuntu@$frontend_ip:~/workers.txt
scp $ssh_opts $frontend_ip_file ubuntu@$frontend_ip:~/frontend.txt
scp $ssh_opts $client_ip_file ubuntu@$frontend_ip:~/clients.txt
scp $ssh_opts $private_frontend_ip_file ubuntu@$frontend_ip:~/private_frontend_ip.txt
scp $ssh_opts $private_worker_ip_file ubuntu@$frontend_ip:~/private_worker_ip.txt
scp $ssh_opts $config_file ubuntu@$frontend_ip:~/config.json

# Create a new bash script containing all of our desired parameters.
cp $args_file background.sh
cat background_base.sh >> background.sh
chmod +x background.sh

# Start running experiment
scp $ssh_opts background.py ubuntu@$frontend_ip:$PARM_REMOTE_HOME/run/
scp $ssh_opts background.sh ubuntu@$frontend_ip:$PARM_REMOTE_HOME/run/
scp $ssh_opts background_client.py ubuntu@$frontend_ip:$PARM_REMOTE_HOME/run/
scp $ssh_opts stop_background.py ubuntu@$frontend_ip:$PARM_REMOTE_HOME/run/
rsync -zh -r -e "ssh -i ${key}" --exclude "*plot*" ../stats ubuntu@$frontend_ip:$PARM_REMOTE_HOME/

echo "All done launching"

echo "Starting"
ssh -n $ssh_opts ubuntu@$frontend_ip "\
  cd ${PARM_REMOTE_HOME}/run; \
  nohup ./background.sh &> ~/runner_log.txt" &
echo "Started background"
