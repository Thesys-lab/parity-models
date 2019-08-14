#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <node_type> <index>"
  exit 1
fi

node_type=$1
idx=$2

ssh $(cat ~/${node_type}s.txt | cut -d' ' -f${idx})
