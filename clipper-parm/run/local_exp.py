"""
This file contains "Models" that can be used for exercising the ParM
deployment scripts locally. Each model simply returns lists of random
floating point values. Models are thus computationally inexpensive and
are unlikely to overload a single machine when multiple of them are run
concurrently.

NOTE: This script should *NOT* be used for performance comparisions. The
script is only used to excercise the general deployment scripts.
"""

import argparse
from clipper_admin import ClipperConnection, DistributedParmDockerContainerManager, ModelInstanceWorker
from clipper_admin.deployers import pytorch as pytorch_deployer
import time
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, batch_size):
        super(Model, self).__init__()

        self.outputs = torch.rand(batch_size, 1000)

    def forward(self, inputs, pool=None):
        return self.outputs


def predict(model, inputs, pool=None):
    with torch.no_grad():
        predictions = model(inputs, pool)
    predictions = predictions.data.cpu().numpy()
    return [p for p in predictions]


model = None
red_model = None


def worker(args):
    print(args)
    if args.w_redundant:
        func = predict
        if args.redundancy_mode == "coded":
            input_type = "floats"
        else:
            input_type = "bytes"
        my_model = red_model
    else:
        func = predict

        input_type = "bytes"
        my_model = model

    model_instance = ModelInstanceWorker(name="example",
                                         input_type=input_type,
                                         func=func,
                                         redundant=args.w_redundant,
                                         query_frontend_hostname=args.w_qf_hostname,
                                         serialization_fn=pytorch_deployer.serialize,
                                         pkgs_to_install=['pillow'],
                                         model=my_model,
                                         gpu=torch.cuda.is_available())

    print("Starting worker on port", args.w_port)
    model_instance.run(args.w_port)


def frontend(args):
    print(args)
    batch_size = args.batch_size
    num_models = args.num_models
    num_redundant_models = args.num_redundant_models
    func = predict
    red_func = predict

    if args.redundancy_mode == "none" or args.redundancy_mode == "equal":
        redundancy_mode = 0
    elif args.redundancy_mode == "coded":
        redundancy_mode = 2
    elif args.redundancy_mode == "cheap":
        redundancy_mode = 3

    if args.queue_mode == "single_queue":
        queue_mode = 0
        single_queue = True
    else:
        # Round robin
        queue_mode = 1
        single_queue = False

    assert len(args.f_ips) == num_models + num_redundant_models
    if len(args.f_ports) != len(args.f_ips):
        assert len(args.f_ports) == 1
        args.f_ports *= len(args.f_ips)

    model_instance_ip_port = []
    red_model_instance_ip_port = []
    for i in range(len(args.f_ips)):
        if i < num_models:
            model_instance_ip_port.append((args.f_ips[i], int(args.f_ports[i])))
        else:
            red_model_instance_ip_port.append((args.f_ips[i], int(args.f_ports[i])))

    client_ip_port = []
    if len(args.f_client_ports) != len(args.f_client_ips):
        assert len(args.f_client_ports) == 1
        args.f_client_ports *= len(args.f_client_ips)
    client_ip_port = [(ip, int(port)) for ip, port in zip(args.f_client_ips, args.f_client_ports)]
    cm = DistributedParmDockerContainerManager(model_instance_ip_port=model_instance_ip_port,
                                                 red_model_instance_ip_port=red_model_instance_ip_port,
                                                 client_ip_port=client_ip_port)
    clipper_conn = ClipperConnection(cm, distributed=True)
    frontend_args = {
        "redundancy_mode": redundancy_mode,
        "queue_mode": queue_mode,
        "num_models": num_models,
        "num_redundant_models": num_redundant_models,
        "batch_size": batch_size,
        "mode": args.f_mode
    }

    clipper_conn.start_clipper(frontend_args=frontend_args)

    if args.redundancy_mode == "coded":
        red_input_type = "floats"
    else:
        red_input_type = "bytes"

    pytorch_deployer.create_endpoint(
            clipper_conn=clipper_conn,
            name="example",
            input_type="bytes",
            func=func,
            pytorch_model=model,
            pkgs_to_install=['pillow'],
            num_replicas=num_models,
            batch_size=batch_size,
            num_red_replicas=num_redundant_models,
            red_func=red_func,
            red_input_type=red_input_type,
            red_pytorch_model=red_model,
            prefer_original=False,
            slo_micros=10000000 * 10)

    sleep_time = 5
    print("Sleeping for", sleep_time, "seconds to let things start up")
    time.sleep(sleep_time)

    total_time = cm.run_clients()
    print(total_time)

    with open(args.f_outfile, 'w') as outfile:
        outfile.write("{:.4f}".format(total_time))

    clipper_conn.stop_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str,choices=["frontend", "worker"],
                        help="Whether the caller of this script is a frontend or worker.")
    parser.add_argument("redundancy_mode", type=str, choices=["none", "equal", "coded", "cheap"],
                        help="Redundancy technique to employ")
    parser.add_argument("queue_mode", type=str, choices=["single_queue", "rr"],
                        help="Load-balancing strategy to use.")
    parser.add_argument("model", type=str, choices=["resnet18", "resnet152"],
                        help="Model architecture to use")
    parser.add_argument("--num_models", type=int,
                        help="Number of model replicas that will be launched")
    parser.add_argument("--num_redundant_models", type=int,
                        help="Number of redundant models that will be launched")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size to be enforced by the model server")
    parser.add_argument("--f_ips", nargs='+',
                        help="List of worker ip addresses for frontend to contact")
    parser.add_argument("--f_ports", nargs='+',
                        help="List of worker ports for frontend to contact")
    parser.add_argument("--f_client_ips", nargs='+',
                        help="List of client ip addresses for frontend to contact")
    parser.add_argument("--f_client_ports", nargs='+',
                        help="List of client ports for frontend to contact")
    parser.add_argument("--f_mode", type=str,
                        help="Which mode {debug, release} to run in")
    parser.add_argument("--f_outfile", type=str,
                        help="File to write stats from runner to")
    parser.add_argument("--w_port", type=int,
                        help="Worker port to listen on")
    parser.add_argument("--w_redundant", action="store_true",
                        help="Whether worker is redundant")
    parser.add_argument("--w_qf_hostname", type=str,
                        help="Query frontend hostname")
    args = parser.parse_args()

    model = Model(args.batch_size)
    red_model = Model(args.batch_size)

    if args.task == "frontend":
        frontend(args)
    elif args.task == "worker":
        worker(args)
    else:
        assert False, "Unrecognized task '{}'".format(args.task)

