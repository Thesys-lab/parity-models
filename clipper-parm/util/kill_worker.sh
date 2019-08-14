#!/bin/bash

for port in 7000; do
  pid=$(lsof -i :$port | grep TCP | awk -F' ' '{print $2}')
  kill $pid &> /dev/null
done
