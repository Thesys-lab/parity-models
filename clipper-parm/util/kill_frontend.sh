#!/bin/bash

for port in 1477; do
  pid=$(lsof -i :$port | grep TCP | awk -F' ' '{print $2}')
  kill $pid &> /dev/null
done
