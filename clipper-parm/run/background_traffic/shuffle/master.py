import argparse
import os
import random
import socket
import struct
from threading import Thread

import common


stop = False


def start_shuffle_at_worker(sender_ip, receiver_idx, mb_to_send):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Connecting to", sender_ip, common.WORKER_MGMT_PORT)
    sock.connect((sender_ip, common.WORKER_MGMT_PORT))
    print("sending to worker", receiver_idx, mb_to_send, flush=True)
    sock.send(common.master_to_snd_format_struct.pack(*(receiver_idx, mb_to_send,)))

    unpacker = struct.Struct('I')
    data = sock.recv(unpacker.size)
    assert unpacker.unpack(data)[0] == 0
    sock.close()


def run(args):
    global stop
    always_select_ips = args.worker_ips[:args.num_model_instances]
    total_ips = args.worker_ips

    inference_indices = list(range(args.num_model_instances))
    while not stop:
        mb_to_send = random.randint(128, 256)
        print(mb_to_send)
        #w0 = random.randint(0, len(always_select_ips)-1)
        w0 = random.randint(0, len(total_ips)-1)
        w1 = random.randint(0, len(total_ips)-1)


        group0 = w0 // args.num_model_instances
        group1 = w1 // args.num_model_instances
        # Make sure we don't shuffle to self, and don't shuffle between two of the same group
        while (w0 == w1) and len(args.worker_ips) > 1: #while (group0 == group1) and len(args.worker_ips) > 1:
            w1 = random.randint(0, len(total_ips)-1)
            group1 = w1 // args.num_model_instances

        if random.random() > 0.5:
            sender = w0
            receiver = w1
        else:
            sender = w1
            receiver = w0
        #sender = w1
        #receiver = w0

        sender_ip = total_ips[sender]
        receiver_ip = total_ips[receiver]
        print(sender, "({})".format(sender_ip), "->", receiver, "({})".format(receiver_ip))
        start_shuffle_at_worker(sender_ip, int(receiver), int(mb_to_send))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num_outstanding", type=int, help="Number of outstanding shuffles")
    parser.add_argument("num_model_instances", type=int, help="Number of model-serving instances")
    parser.add_argument("--worker_ips", nargs='+', help="ip addresses of workers")
    args = parser.parse_args()

    print(args, flush=True)
    #random.seed(42)

    assert len(args.worker_ips) >= args.num_model_instances

    with open("/tmp/shuffle_pid.txt", 'w') as outfile:
        outfile.write(str(os.getpid()))

    threads = []
    for i in range(args.num_outstanding):
        t = Thread(target=run, args=(args,))
        t.start()
        threads.append(t)

    # Listen to a ud socket to determine when we should quit.
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind('/home/ubuntu/bg_sock')
    sock.listen(5)
    (clientsocket, address) = sock.accept()
    print("Stopping all master threads")
    stop = True

    for t in threads:
        t.join()

    print("Master threads joined")
    print("Sending response", flush=True)
    clientsocket.sendall('1'.encode())
    clientsocket.close()
    sock.close()

    # Tell each 
