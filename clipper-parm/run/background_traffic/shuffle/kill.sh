#!/bin/bash

#kill $(ps -aux | grep "python3 worker.py" | head -n 1 | awk -F' ' '{ print $2 }')
python3 stop_background.py
