#!/bin/bash

container=34a13fe49038
docker start $container
docker attach $container
