import argparse
import base64
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
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


def request(idx):
    global img_data, ip_addr
    start = time.time()
    url = "http://{}:1337/example/predict".format(ip_addr)
    req_json = json.dumps({
        "input": img_data[idx]
    })
    headers = {'Content-type': 'application/json'}
    print("sent={}".format(time.time()))
    r = requests.post(url, headers=headers, data=req_json)
    end = time.time()
    print("time={}".format(end - start))
    return (end - start)


def batch_request(indices):
    threads = []
    for i in indices:
        t = Thread(target=request, args=(i,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


def group_request(indices):
    procs = []
    for i in indices:
        p = Process(target=batch_request, args=(i,))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


def grouped_requests(num_groups, num_batches_per_group, num_groups_outstanding,
                     batch_size, num_imgs):
    global img_data
    queries = []
    for _ in range(num_groups):
        indices = []
        for _ in range(num_batches_per_group):
            indices.append(
                [random.randrange(num_imgs) for _ in range(batch_size)])
        queries.append(indices)

    pool = ThreadPoolExecutor(max_workers=num_groups_outstanding)
    start = time.time()
    for _ in pool.map(group_request, queries):
        continue
    end = time.time()
    pool.shutdown()
    return (end-start)


q = queue.Queue()
# Modified from https://stackoverflow.com/questions/16914665/how-to-use-queue-with-concurrent-future-threadpoolexecutor-in-python-2
def prepare_queries(query_info):
    for query in query_info:
        time.sleep(query[0])
        q.put(query[1])
    return "DONE FEEDING"


def poisson_individual_requests(args, num_imgs):
    global img_data
    total_num_queries = args.num_groups * args.num_batches_per_group * args.batch_size
    queries = []

    interarrival_times = np.random.exponential(1./args.rate, total_num_queries)

    for i in range(total_num_queries):
        queries.append((interarrival_times[i], random.randrange(num_imgs)))

    max_num_outstanding = 10 * args.rate
    reqs = []
    with ThreadPoolExecutor(max_workers=max_num_outstanding) as executor:
        start = time.time()
        for st, idx in queries:
            time.sleep(st)
            reqs.append(executor.submit(request, idx))

    end = time.time()
    return (end-start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_groups", type=int,
                        help="Total number of encoding groups to send")
    parser.add_argument("--num_groups_outstanding", type=int,
                        help="Allowed outstanding groups")
    parser.add_argument("--num_batches_per_group", type=int,
                        help="Number batches per encoding groups")
    parser.add_argument("--batch_size", type=int,
                        help="Number of queries per batch")
    parser.add_argument("--frontend_ip", type=str,
                        help="IP address of frontend")
    parser.add_argument("--frontend_private_ip", type=str,
                        help="Private IP address of frontend")
    parser.add_argument("--port", type=int,
                        help="Port to listen on")
    parser.add_argument("--img_dir", type=str,
                        help="Path to directory containing images")
    parser.add_argument("--num_imgs", type=int,
                        help="Number of images that can be queried")
    parser.add_argument("--frontend_port", type=int, default=1477,
                        help="Port used by frontend")
    parser.add_argument("--rate", type=int,
                        help="Average # queries to send per second. "
                             "Used for generating a Poisson process.")
    args = parser.parse_args()
    print(args)

    assert os.path.isdir(args.img_dir)
    ip_addr = args.frontend_ip
    files = os.listdir(args.img_dir)
    imgs = [os.path.join(args.img_dir, im) for im in files if "jpg" in im or "jpeg" in im]
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
    sock.close()
    assert msg == MSG_START, "Unexpected msg at start {}".format(msg)


    # Run grouped requests as warmups
    # If you change the number used for warmup, you must change
    # $CLIPPER_HOME/stats/get_metrics_logs.sh
    num_warmup = 5
    print("START WARMUP")
    for _ in range(num_warmup):
        grouped_requests(1, args.num_batches_per_group,
                         1, args.batch_size, num_imgs)
    print("DONE WARMUP")

    elapsed_time = poisson_individual_requests(args, num_imgs)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Connecting to", args.frontend_private_ip, args.frontend_port)
    sock.connect((args.frontend_private_ip, args.frontend_port))
    sock.send(msg_packer.pack(*(MSG_STOP,)))
    sock.close()

    print("Done")
    num_queries = args.num_groups * args.num_batches_per_group * args.batch_size
    throughput = num_queries / elapsed_time

    print("Client-viewed throughput")
    print(num_queries, "/", elapsed_time, "=", throughput)
