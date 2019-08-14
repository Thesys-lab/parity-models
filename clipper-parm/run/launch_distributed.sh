#!/bin/bash

NUMARG=15
if [ "$#" -ne "$NUMARG" ]; then
  echo "Usage: $0 <exp_id> <red_mode> <queue_mode> <ec_k_val> <num_models>"
  echo "          <num_red_models> <frontend_type> <worker_type> <client_type>"
  echo "          <num_clients> <build_mode> <reuse> <final_dir> <config_file>"
  echo "          <launch_bg>"
  echo "  exp_id:           Identifier of this experiment in tagging ec2 resources."
  echo "                    If reusing, must set reuse=1"
  echo "  red_mode:         One of {none, equal, coded, cheap}"
  echo "  queue_mode:       One of {single_queue, rr}"
  echo "  ec_k_val:         K value to use in coding"
  echo "  num_models:       Number of worker instances to launch (both parity and not)"
  echo "  num_red_models:   Number of redundant worker instances to launch."
  echo "  frontend_type:    Type of instance to use for frontend"
  echo "  worker_type:      Type of instance to use for worker"
  echo "  client_type:      Type of instance to use for client"
  echo "  num_clients:      Number of clients to launch"
  echo "  build_mode:       One of {debug, release}"
  echo "  reuse:            Whether to reuse existing exp_id"
  echo "  final_dir:        Final directory to save to"
  echo "  config_file:      File containing json configuration"
  echo "  launch_bg:        Whether to launch a background copy of Clipper"
  exit 1
fi

exp_id=$1
red_mode=$2
queue_mode=$3
ec_k_val=$4
num_models=$5
num_red_models=$6
frontend_type=$7
worker_type=$8
client_type=$9
num_clients=${10}
build_mode=${11}
reuse=${12}
final_dir=${13}
config_file=${14}
launch_bg=${15}

if [ ! -d cur_exp ]; then
  mkdir cur_exp
fi

if [ "$build_mode" != "debug" ] && [ "$build_mode" != "release" ]; then
  echo "Unrecognized build mode ${build_mode}"
  exit 1
fi

root_save_dir=/tmp/$final_dir
mkdir -p $root_save_dir

copy_dir=$root_save_dir/0
mkdir $copy_dir
exp_id=${exp_id}
my_dir=$copy_dir
clipper_idx=0

total_num_models=$num_models
echo "total_num_models is ${total_num_models}"

reuse_flag=--reuse
if [ "$reuse" -eq "0" ]; then
  reuse_flag=''
fi

frontend_ip_file=$my_dir/frontend_ips.txt
worker_ip_file=$my_dir/worker_ips.txt
client_ip_file=$my_dir/client_ips.txt

all_worker_ip_file=$my_dir/worker_ips_all.txt
private_frontend_ip_file=$my_dir/private-master_ips.txt
private_worker_ip_file=$my_dir/private-worker_ips.txt

# Create frontend instances
python3 launch_ec2.py \
  ${exp_id} frontend 1 ${red_mode} \
  --instance_type ${frontend_type} \
  --save_ip_file ${frontend_ip_file} \
  --save_private_ip_file ${private_frontend_ip_file} \
  --save_instance_ids_file cur_exp/frontend_ids.txt \
  ${reuse_flag} #--no_spot

# Create worker instances
python3 launch_ec2.py \
  ${exp_id} worker ${total_num_models} ${red_mode} \
  --red_models ${num_red_models} \
  --instance_type ${worker_type} \
  --save_ip_file ${worker_ip_file} \
  --save_private_ip_file ${private_worker_ip_file} \
  --save_instance_ids_file cur_exp/worker_ids.txt \
  ${reuse_flag}

# Create client instances
python3 launch_ec2.py \
  ${exp_id} client ${num_clients} ${red_mode} \
  --instance_type ${client_type} \
  --save_ip_file ${client_ip_file} \
  --save_instance_ids_file cur_exp/client_ids.txt \
  ${reuse_flag}

sleep 30

cp $frontend_ip_file cur_exp/frontend_ip.txt

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
key_name=$my_dir/id_rsa_${clipper_idx}.pub
scp $ssh_opts ubuntu@$frontend_ip:~/.ssh/id_rsa.pub $key_name

# Copy key to workers and clients
for wip in $all_worker_ips $client_ips $frontend_ip; do
  ssh-keygen -R $wip >/dev/null
  scp $ssh_opts $key_name ubuntu@$wip:~/id_rsa.pub &
done
wait

for wip in $all_worker_ips $client_ips $frontend_ip; do
  ssh $ssh_opts ubuntu@$wip "cat id_rsa.pub >> ~/.ssh/authorized_keys" &
done
wait

PARM_REMOTE_HOME=/home/ubuntu/clipper-parm

# Rsync clipper source to frontend
config_flag=--release
if [ "$build_mode" != "release" ]; then
  config_flag=''
fi

rsync -zh -r -t -e "ssh -i ${key}" $PARM_LOCAL_HOME/ ubuntu@$frontend_ip:$PARM_REMOTE_HOME/
ssh $ssh_opts ubuntu@$frontend_ip "\
  cd ${PARM_REMOTE_HOME}; \
  if [ ! -d ${build_mode} ]; then ./configure ${config_flag}; fi; \
  cd ${build_mode}; \
  make -j4"

# Rsync clipper_admin to frontend and all workers, along with background traffic
echo "All worker ips are: ${all_worker_ips}"
for ip in $frontend_ip $all_worker_ips $(cat $client_ip_file); do
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

# Start running experiment
scp $ssh_opts start_experiment_from_frontend.sh ubuntu@$frontend_ip:$PARM_REMOTE_HOME/run/
scp $ssh_opts pytorch_distributed.py ubuntu@$frontend_ip:$PARM_REMOTE_HOME/run/
scp $ssh_opts client.py ubuntu@$frontend_ip:$PARM_REMOTE_HOME/run/
rsync -zh -r -e "ssh -i ${key}" --exclude "*plot*" ../stats ubuntu@$frontend_ip:$PARM_REMOTE_HOME/

echo "Copying multiple_run_exp.py to frontend"
scp $ssh_opts multiple_run_exp.py ubuntu@$frontend_ip:$PARM_REMOTE_HOME/run/

echo "All done launching"

echo "Putting all ips together"
global_worker_public_ips=$root_save_dir/all_worker_public_ips.txt
cp $root_save_dir/0/worker_ips.txt $global_worker_public_ips

global_worker_private_ips=$root_save_dir/all_worker_private_ips.txt
cp $root_save_dir/0/private-worker_ips.txt $global_worker_private_ips

echo "Public ips are $(cat $global_worker_public_ips)"
echo "Private ips are $(cat $global_worker_private_ips)"

master_frontend_ip=$(cat $root_save_dir/0/frontend_ips.txt)
echo "Copying private ips to all workers"
for ip in $master_frontend_ip $(cat $global_worker_public_ips); do
  scp $ssh_opts $global_worker_private_ips ubuntu@$ip:~/private_worker_ip.txt &
done
wait
echo "Copied all private ips"

if [ "$launch_bg" -gt 0 ]; then
  echo "Launching background instances"
  python3 launch_background.py $config_file
fi

echo "Starting run on frontend"
copy_dir=$root_save_dir/0
frontend_ip=$(cat $copy_dir/frontend_ips.txt)
my_final_dir=$final_dir/0
ssh -n $ssh_opts ubuntu@$frontend_ip "\
  cd ${PARM_REMOTE_HOME}/run; \
  nohup python3 run_exp.py ~/config.json run --final_dir ${my_final_dir} --idx 0 &> ~/runner_log.txt" &
wait

echo "Finished"
echo "Final directory is ${final_dir}"
