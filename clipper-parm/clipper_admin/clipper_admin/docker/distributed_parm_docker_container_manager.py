from __future__ import absolute_import, division, print_function
import docker
import logging
import os
import sys
import random
import time
import json
import socket
import struct
import threading
import time

from ..container_manager import (
    create_model_container_label, parse_model_container_label,
    ContainerManager, CLIPPER_DOCKER_LABEL, CLIPPER_MODEL_CONTAINER_LABEL,
    CLIPPER_QUERY_FRONTEND_CONTAINER_LABEL,
    CLIPPER_MGMT_FRONTEND_CONTAINER_LABEL, CLIPPER_INTERNAL_RPC_PORT,
    CLIPPER_INTERNAL_QUERY_PORT, CLIPPER_INTERNAL_MANAGEMENT_PORT,
    CLIPPER_INTERNAL_METRIC_PORT)
from ..exceptions import ClipperException
from requests.exceptions import ConnectionError
from .docker_metric_utils import *
from .parm_docker_container_manager import ParmDockerContainerManager
from .common import *

logger = logging.getLogger(__name__)

class DistributedParmDockerContainerManager(ParmDockerContainerManager):
    def __init__(self,
                 model_instance_ip_port,
                 red_model_instance_ip_port,
                 client_ip_port,
                 docker_ip_address="localhost",
                 clipper_query_port=1337,
                 clipper_management_port=1338,
                 clipper_rpc_port=7000,
                 redis_ip=None,
                 redis_port=6379,
                 prometheus_port=9090,
                 docker_network="clipper_network",
                 location="/home/ubuntu/clipper-parm/",
                 extra_container_kwargs={}):
        super().__init__(docker_ip_address,
                         clipper_query_port,
                         clipper_management_port,
                         clipper_rpc_port,
                         redis_ip,
                         redis_port,
                         prometheus_port,
                         docker_network,
                         location,
                         extra_container_kwargs)
        self.model_instance_ip_port = model_instance_ip_port
        self.red_model_instance_ip_port = red_model_instance_ip_port
        self.client_ip_port = client_ip_port

    def set_num_replicas(self, name, version, input_type, image, num_replicas, redundant=False, volumes=None, cmd=None):
        if redundant:
            hosts = self.red_model_instance_ip_port
        else:
            hosts = self.model_instance_ip_port
        self.send_command_to(hosts, MSG_START)

    def stop_all(self):
        print("Sending stop_all")
        self.send_command_to(self.model_instance_ip_port + self.red_model_instance_ip_port, MSG_STOP)
        print("Got all stop_all responses")
        super().stop_all()

    def stop_all_clients(self):
        self.send_command_to(self.client_ip_port, MSG_STOP, wait_response=True)

    def send_command_to(self, hosts, msg, wait_response=True):
        threads = []
        for ip, port in hosts:
            t = threading.Thread(target=self.command_model_at_host, args=(ip, port, msg, wait_response))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def get_num_replicas(self, name, version):
        return len(self.model_instance_ip_port)

    def command_model_at_host(self, ip, port, msg, wait_response):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Connecting to", ip, port, flush=True)
        try:
            sock.connect((ip, port))
            sock.send(msg_packer.pack(*(msg,)))
            if wait_response:
                print("Waiting for response", flush=True)
                wait = sock.recv(msg_packer.size)
                assert msg_packer.unpack(wait)[0] == msg, "Received unknown response {}".format(msg_packer.unpack(wait)[0])
            else:
                print("Not waiting for response", flush=True)
            sock.close()
        except:
            print("Error connecting to", ip, port, flush=True)
            raise

    def run_clients(self, wait=True):
        num_clients = len(self.client_ip_port)
        print(self.client_ip_port)
        print("frontend num_clients is", num_clients, flush=True)
        start_time = time.time()
        self.send_command_to(self.client_ip_port, MSG_START, wait_response=False)

        if wait:
            num_done = 0
            ip = "0.0.0.0"
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((ip, 1477))
            sock.listen(num_clients)

            while num_done < num_clients:
                (clientsocket, address) = sock.accept()
                print("Got connection", flush=True)
                data = clientsocket.recv(msg_packer.size)
                msg = msg_packer.unpack(data)[0]
                assert msg == MSG_STOP, "Unexpected message {}".format(msg)
                num_done += 1
                print("Have", num_done, "/", num_clients, flush=True)
                clientsocket.close()

            sock.close()
            end_time = time.time()
            return end_time - start_time
