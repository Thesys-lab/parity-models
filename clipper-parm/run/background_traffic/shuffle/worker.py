import argparse
import multiprocessing
import os
import random
import socket
import struct
import time
from threading import Thread
import torch

import common


stop = False


def tight_loop():
    while True:
        for i in range(1000000):
            x = i * i


def tight_loop_torch():
    num_mats = 10
    mat_dim = 512
    x = torch.rand(num_mats, mat_dim, mat_dim)
    if torch.cuda.is_available():
        x = x.cuda()

    i = 0
    while True:
        next_idx = (i + 1) % num_mats
        next_plus_one_idx = (i + 2) % num_mats
        x[i] = x[next_idx] * x[next_plus_one_idx]
        i = next_idx


def recv_bytes(sock):
    data = sock.recv(common.snd_to_recv_format_struct.size)
    num_mb_recv = common.snd_to_recv_format_struct.unpack(data)[0]

    total_bytes_recvd = 0
    total_bytes_to_recv = num_mb_recv * 1024 * 1024
    print("Start recv", num_mb_recv, "MB")
    while total_bytes_recvd < total_bytes_to_recv:
        buf = sock.recv(total_bytes_to_recv - total_bytes_recvd)
        total_bytes_recvd += len(buf) 
    return num_mb_recv


def recv(sock):
    # Receive bytes from sender
    num_mb = recv_bytes(sock) 

    """sleep_time = random.uniform(0, 2)#4)#random.randint(1, 2)
    processes = []
    for i in range(2):#multiprocessing.cpu_count()):
        p = multiprocessing.Process(target=tight_loop_torch)#tight_loop)
        processes.append(p)
        p.start()

    print("processing for", sleep_time)
    time.sleep(sleep_time)
    for p in processes:
        p.terminate()"""

    # Shuffle back to sender
    send_bytes(sock, num_mb)

    sock.close()
    print("Done recv", flush=True)


def listen_recv(args):
    global stop

    ip = "0.0.0.0"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((ip, common.WORKER_RECV_PORT))
    sock.listen(args.num_outstanding)

    while not stop:
        sock.settimeout(5)
        try:
            (clientsocket, _) = sock.accept()
            t = Thread(target=recv, args=(clientsocket,))
            t.start()
        except socket.timeout:
            print("Receiver timeout, stop={}".format(stop))
            pass

    print("Receiver got stop signal", flush=True)
    # Sleep for some time to make sure any current recvs finish
    time.sleep(10)
    sock.close()


def listen_send(args):
    global stop

    print("starting listen_thread", flush=True)
    ip = "0.0.0.0"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((ip, common.WORKER_MGMT_PORT))
    sock.listen(args.num_outstanding)
    print("started listening on port {}".format(common.WORKER_MGMT_PORT))

    while not stop:
        sock.settimeout(5)
        try:
            (clientsocket, _) = sock.accept()
            t = Thread(target=send, args=(clientsocket, args.worker_ips))
            t.start()
        except socket.timeout:
            print("Sender timeout, stop={}".format(stop), flush=True)
            pass

    print("Sender got stop signal", flush=True)
    # Sleep for some time to make sure any current sends finish
    time.sleep(10)
    sock.close()

def send_bytes(sock, num_mb_send):
    buf = bytearray(65536)
    sock.send(common.snd_to_recv_format_struct.pack(*(num_mb_send,)))

    total_bytes_sent = 0
    total_bytes_to_send = num_mb_send * 1024 * 1024
    print("Sending", num_mb_send, "MB")
    while total_bytes_sent < total_bytes_to_send:
        bytes_to_send = min(total_bytes_to_send - total_bytes_sent, len(buf))
        bytes_sent = sock.send(buf[:bytes_to_send])
        total_bytes_sent += bytes_sent


def send(clientsocket, worker_ips):
    data = clientsocket.recv(common.master_to_snd_format_struct.size)
    unpacked = common.master_to_snd_format_struct.unpack(data)

    worker_ip = worker_ips[unpacked[0]]
    num_mb_send = unpacked[1]

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Connecting to", worker_ip, common.WORKER_RECV_PORT)
    sock.connect((worker_ip, common.WORKER_RECV_PORT))

    # Send bytes to receiver
    send_bytes(sock, num_mb_send)

    # Receive bytes back from receive
    recv_bytes(sock)
    sock.close()

    clientsocket.send(struct.pack('I', 0))
    clientsocket.close()
    print("Done sending")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num_outstanding", type=int, help="Number of possible outstanding shuffles")
    parser.add_argument("--worker_ips", nargs='+', help="ip addresses of workers")
    args = parser.parse_args()

    #random.seed(42)

    with open("/tmp/shuffle_pid.txt", 'w') as outfile:
        outfile.write(str(os.getpid()))

    listen_recv_thread = Thread(target=listen_recv, args=(args,))
    listen_recv_thread.start()

    listen_send_thread = Thread(target=listen_send, args=(args,))
    listen_send_thread.start()

    print("Listening on udf socket", flush=True)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock_name = '/home/ubuntu/bg_sock'
    sock.bind(sock_name)
    sock.listen(5)
    (clientsocket, address) = sock.accept()
    print("Stopping all worker threads", flush=True)
    stop = True
    listen_recv_thread.join()
    listen_send_thread.join()

    print("All worker threads joined", flush=True)
    print("Sending response", flush=True)
    clientsocket.sendall('1'.encode())
    clientsocket.close()
    sock.close()
