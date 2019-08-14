#!/bin/bash

terminate_if_exists() {
  fil=$1
  if [ -f $fil ]; then
    aws ec2 terminate-instances --instance-ids $(cat $fil)
  fi
}

terminate_if_exists cur_exp/frontend_ids.txt
terminate_if_exists cur_exp/worker_ids.txt
terminate_if_exists cur_exp/client_ids.txt
terminate_if_exists cur_exp/bg_frontend_ids.txt
terminate_if_exists cur_exp/bg_client_ids.txt
