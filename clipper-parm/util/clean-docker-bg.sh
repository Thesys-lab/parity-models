#!/bin/bash

for name in redis-parm query_frontend-parm mgmt_frontend-parm query_frontend_exporter-parm metric_frontend-parm; do
    docker kill $name
    docker rm $name
done

remove_by_keyword() {
  images=$(docker ps -a | grep "$1" | awk -F' ' '{print $2}')
  containers=$(docker ps -a | grep "$1" | awk -F' ' '{print $1}')
  if [ "$containers" != "" ]; then
    echo "Removing containers with $1"
    docker kill $containers
    docker rm $containers
  fi

  if [ "$images" != "" ]; then
    echo "Removing images with $1"
    docker image rm $images
  fi
}

remove_by_keyword bg
remove_by_keyword /bin/sh

dangling_volumes=$(docker volume ls -qf dangling=true)
if [ "$dangling_volumes" != "" ]; then
  echo "Removing dangling volumes"
  docker volume rm $dangling_volumes
fi
