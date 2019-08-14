import argparse
import base64
import concurrent.futures
import json
from multiprocessing import Process
import numpy as np
import os
import queue
import random
import requests
import socket
import struct
from threading import Thread
import time

from clipper_admin.docker.common import *

img_data = []
ip_addr = None
q = queue.Queue()
stop_feeding = False


def request(idx):
    global img_data, ip_addr
    start = time.time()
    url = "http://{}:1337/bg/predict".format(ip_addr)
    print(url)
    req_json = json.dumps({
        "input": img_data[idx]
    })
    headers = {'Content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=req_json)
    end = time.time()
    print(end - start)
    return (end - start)


# Modified from https://stackoverflow.com/questions/16914665/how-to-use-queue-with-concurrent-future-threadpoolexecutor-in-python-2
def prepare_queries(query_info):
    global stop_feeding
    while True:
        for query in query_info:
            time.sleep(query[0])
            q.put(query[1])

            if stop_feeding:
                print("Returning from prepare_queries", flush=True)
                return "DONE FEEDING"
    return "DONE FEEDING"


def poisson_individual_requests(args, num_imgs):
    global img_data
    global stop_feeding
    total_num_queries = 100000
    queries = []

    sleep_times = np.random.exponential(1./args.rate, total_num_queries)
    #sleep_times = [1. / args.rate] * total_num_queries

    send_times = [sleep_times[0]]
    for i in range(1, len(sleep_times)):
        send_times.append(sleep_times[i] + send_times[i-1])

    for i in range(total_num_queries):
        queries.append((send_times[i], random.randrange(num_imgs)))

    max_num_outstanding = 30#args.rate#10*args.rate #100
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_num_outstanding) as executor:
        while not stop_feeding:
            reqs = []
            start = time.time()
            for st, idx in queries:
                progress = time.time() - start
                if progress < st:
                    time.sleep(st - progress)
                reqs.append(executor.submit(request, idx))

                if stop_feeding:
                    break

    print("Exiting main loop", flush=True)
    end = time.time()
    return (end-start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend_ip", type=str, help="IP address of frontend")
    parser.add_argument("--port", type=int, help="Port to listen on")
    parser.add_argument("--img_dir", type=str, help="Path to directory containing images")
    parser.add_argument("--num_imgs", type=int, help="Number of images that can be queried")
    parser.add_argument("--rate", type=int, help="Average # queries to send per second. Used for generating a Poisson process.")
    args = parser.parse_args()
    print(args)

    assert os.path.isdir(args.img_dir)
    ip_addr = args.frontend_ip
    imgs = [os.path.join(args.img_dir, im) for im in os.listdir(args.img_dir) if "jpg" in im]
    num_imgs = min(args.num_imgs, len(imgs))
    imgs = imgs[:num_imgs]

    for img in imgs:
        with open(img, 'rb') as infile:
            img_data.append(base64.b64encode(infile.read()).decode())

    ip = "0.0.0.0"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((ip, args.port))
    sock.listen(5)

    (clientsocket, addr) = sock.accept()
    print("addr is", addr)
    data = clientsocket.recv(msg_packer.size)
    msg = msg_packer.unpack(data)[0]
    clientsocket.send(msg_packer.pack(*(msg,)))
    clientsocket.close()
    assert msg == MSG_START, "Unexpected msg at start '{}'".format(msg)

    # elapsed_time = poisson_individual_requests(args, num_imgs)
    req_thread = Thread(target=poisson_individual_requests, args=(args, num_imgs))
    req_thread.start()

    print("Listening for another connection", flush=True)
    (clientsocket, addr) = sock.accept()
    data = clientsocket.recv(msg_packer.size)
    msg = msg_packer.unpack(data)[0]
    assert msg == MSG_STOP, "Unexpected msg at stop '{}'".format(msg)
    print("Setting stop_feeding to true", flush=True)
    stop_feeding = True

    print("Waiting for thread to join", flush=True)
    req_thread.join()
    print("Thread joined", flush=True)

    clientsocket.send(msg_packer.pack(*(msg,)))
    clientsocket.close()
    sock.close()

    print("done")
