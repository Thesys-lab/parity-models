import argparse
from clipper_admin import ClipperConnection, DistributedParmDockerContainerManager, ModelInstanceWorker
from clipper_admin.deployers import pytorch as pytorch_deployer
import io
import requests
import time
import socket
import torch
import torch.nn as nn
import torchvision.models
import torchvision.transforms as transforms
from PIL import Image


class ModelWrapper(nn.Module):
    def __init__(self, underlying_model, batch_size):
        super(ModelWrapper, self).__init__()
        self.model = underlying_model
        self.model.eval()
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])

        self.inputs = torch.zeros(batch_size, 3, 224, 224) #torch.rand(batch_size, 3, 224, 224)
        if torch.cuda.is_available():
            print("cuda available")

        self.outputs = self.outputs = torch.autograd.Variable(torch.rand(1, 10))

    def preprocess(self, in_bytes):
        im = Image.open(io.BytesIO(in_bytes))
        #im = torch.rand(3, 224, 224)
        return self.transform(im)

    def forward(self, inputs):
        for i in range(self.batch_size):
            self.inputs[i] = self.preprocess(inputs[i])

        model_in = self.inputs
        if torch.cuda.is_available():
            model_in = model_in.cuda()

        return self.model(model_in)


def predict(model, inputs, pool=None):
    with torch.no_grad():
        predictions = model(inputs).data.cpu().numpy()
    return [p for p in predictions]

model = None
red_model = None
base_port = 7001

def worker(args):
    func = predict
    input_type = "bytes"
    my_model = model

    print("Worker", args.w_idx, "booting up", flush=True)
    model_instance = ModelInstanceWorker(name="bg",
                                         input_type=input_type,
                                         func=func,
                                         redundant=False,
                                         query_frontend_hostname=args.w_qf_hostname,
                                         serialization_fn=pytorch_deployer.serialize,
                                         pkgs_to_install=['pillow'],
                                         model=my_model,
                                         gpu=torch.cuda.is_available())

    # Calculate the port on which this worker should listen.
    port = base_port + (args.w_idx // args.num_workers)
    print("Starting worker on port", port, flush=True)
    model_instance.run(port)


def frontend(args):
    print(args)
    batch_size = args.batch_size
    num_models = args.num_models
    func = predict
    red_func = predict

    assert args.redundancy_mode in ["none", "equal"]
    redundancy_mode = 0

    if args.queue_mode == "single_queue":
        queue_mode = 0
    elif args.queue_mode == "rr":
        queue_mode = 1
    else:
        assert False, "Unrecognized queue mode '{}'".format(args.queue_mode)

    model_instance_ip_port = []
    red_model_instance_ip_port = []
    cur_port = base_port
    if num_models < len(args.f_ips):
	# Round up to highest int so as not to launch more models than needed.
        num_between = int(len(args.f_ips) / num_models + 0.5)
        chosen_indices = list(range(0, len(args.f_ips), num_between))
        print("Range is", chosen_indices)

        # Shift our chosen indices so that they are evenly distributed
        # throughout the clients.
        delta = len(args.f_ips) - chosen_indices[-1]
        shift = delta // 2
        if len(args.f_ips) == 15:
            shift += 1
        chosen_indices = [i + shift for i in chosen_indices]
        print("Shifted range is", chosen_indices)
        for i in chosen_indices:
            model_instance_ip_port.append((args.f_ips[i], cur_port))

    else:
        for i in range(num_models):
            model_instance_ip_port.append(
                    (args.f_ips[i % len(args.f_ips)], cur_port))

            # Wrap around to the next port number if we will ned to repeat workers.
            if i % len(args.f_ips) == len(args.f_ips) - 1:
                cur_port += 1

    print("Model instance ip, port:", model_instance_ip_port)
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
        "num_redundant_models": 0,
        "batch_size": batch_size,
        "mode": args.f_mode,
    }

    clipper_conn.start_clipper(frontend_args=frontend_args)

    red_input_type = "bytes"
    pytorch_deployer.create_endpoint(
            clipper_conn=clipper_conn,
            name="bg",
            input_type="bytes",
            func=func,
            pytorch_model=model,
            pkgs_to_install=['pillow'],
            num_replicas=num_models,
            batch_size=batch_size,
            num_red_replicas=0,
            red_func=red_func,
            red_input_type=red_input_type,
            red_pytorch_model=red_model,
            prefer_original=False,
            slo_micros=10000000 * 10)

    sleep_time = 5
    print("Sleeping for", sleep_time, "seconds to let things start up")
    time.sleep(sleep_time)

    cm.run_clients(wait=False)

    # Listen to a ud socket to determine when we should quit.
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind('/home/ubuntu/bg_sock')
    sock.listen(5)
    (clientsocket, address) = sock.accept()

    print("Stopping all clients")
    cm.stop_all_clients()

    print("Sending response")
    clientsocket.sendall('1'.encode())
    clientsocket.close()
    sock.close()

    print("Stopping all")
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
    parser.add_argument("--num_workers", type=int,
                        help="Number of workers that will be launched")
    parser.add_argument("--batch_size", type=int,
                        help="Batch size to be enforced by the model server")
    parser.add_argument("--f_ips", nargs='+',
                        help="List of worker ip addresses for frontend to contact")
    parser.add_argument("--f_client_ips", nargs='+',
                        help="List of client ip addresses for frontend to contact")
    parser.add_argument("--f_client_ports", nargs='+',
                        help="List of client ports for frontend to contact")
    parser.add_argument("--f_mode", type=str,
                        help="Which mode {debug, release} to run in")
    parser.add_argument("--w_idx", type=int, help="Index of this worker.")
    parser.add_argument("--w_qf_hostname", help="Query frontend hostname")
    args = parser.parse_args()

    if args.model == "resnet18":
        underlying_model = torchvision.models.resnet18
    elif args.model == "resnet152":
        underlying_model = torchvision.models.resnet152

    model = ModelWrapper(underlying_model(), args.batch_size)

    if args.task == "frontend":
        frontend(args)
    elif args.task == "worker":
        worker(args)
