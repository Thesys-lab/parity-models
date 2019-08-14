import argparse
import inspect
import json
import os
from subprocess import call


def start(exp_id, final_dir, frontend_type, worker_type, client_type, model,
          num_workers, num_clients, num_models, batch_size, total_send_rate,
          build_mode="release", reuse=1, img_dir="/home/ubuntu/cat_v_dog/test1",
          num_img=500, red_mode="equal", queue_mode="single_queue"):

    # Save the current arguments to a bash file
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    send_rate = total_send_rate // num_clients
    bg_file = "/tmp/bg_args.sh"
    with open(bg_file, 'w') as outfile:
        print("#!/bin/bash", file=outfile)
        for i in args:
            print("{}={}".format(i, values[i]), file=outfile)
        print("rate={}".format(send_rate), file=outfile)

    save_dir = os.path.join('/tmp/background', exp_id, final_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    call(["./launch_background.sh", exp_id, queue_mode, frontend_type,
          worker_type, client_type, str(num_workers), str(num_clients), str(reuse),
          save_dir, bg_file])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str,
                        help="Path to JSON configuration file")
    args = parser.parse_args()

    with open(args.config_file, 'r') as infile:
        cfg = json.load(infile)

    exp_id = cfg["experiment_id"]
    worker_type = cfg["worker_type"]
    client_type = cfg["client_type"]
    model = cfg["model"]
    num_workers = cfg["num_models"]

    # Launch models on 1/9th of the servers (+ 0.5 is to round to next highest
    # integer)
    num_models = int(round(num_workers / 9 + 0.5))

    # Constant parameters
    final_dir = "bg"
    red_mode = "equal"
    queue_mode = "single_queue"
    frontend_type = "c5.xlarge"
    batch_size = 1
    num_clients = 1
    if worker_type == "c5.xlarge":
        total_send_rate = 16
    elif worker_type == "p2.xlarge":
        total_send_rate = 28
    else:
        assert False, "Uncommon worker_type '{}'".format(worker_type)
    build_mode = "release"
    reuse = 1
    img_dir = "/home/ubuntu/cat_v_dog/test1"
    num_img = 500

    start(exp_id, final_dir, frontend_type, worker_type, client_type, model,
          num_workers, num_clients, num_models, batch_size, total_send_rate,
          build_mode=build_mode, reuse=reuse, img_dir=img_dir,
          num_img=num_img, red_mode=red_mode, queue_mode=queue_mode)
