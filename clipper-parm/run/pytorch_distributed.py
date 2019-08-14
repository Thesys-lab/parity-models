import argparse
from clipper_admin import ClipperConnection, DistributedParmDockerContainerManager, ModelInstanceWorker
from clipper_admin.deployers import pytorch as pytorch_deployer
from concurrent.futures import ThreadPoolExecutor
import io
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torchvision.models
import torchvision.transforms as transforms
from PIL import Image


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    # Adapted from: https://github.com/tonylins/pytorch-mobilenet-v2
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


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

        self.inputs = torch.zeros(batch_size, 3, 224, 224)
        if torch.cuda.is_available():
            print("cuda available")

    def preprocess(self, in_bytes):
        im = Image.open(io.BytesIO(in_bytes))
        return self.transform(im)

    def forward(self, inputs, pool=None):
        if self.batch_size > 1:
            i = 0
            # Preprocess in parallel
            for data in pool.map(self.preprocess, inputs):
                self.inputs[i] = data
                i += 1
        else:
            self.inputs[0] = self.preprocess(inputs[0])

        model_in = self.inputs
        if torch.cuda.is_available():
            model_in = model_in.cuda()

        outs = self.model(model_in)
        return outs


class RedModelWrapper(nn.Module):
    def __init__(self, underlying_model, batch_size):
        super(RedModelWrapper, self).__init__()
        self.model = underlying_model
        self.model.eval()
        self.batch_size = batch_size
        self.inputs = np.empty((batch_size, 3*224*224), dtype='float32')
        if torch.cuda.is_available():
            print("cuda available")

    def forward(self, inputs, pool=None):
        # No need to preprocess as this is done at the frontend.
        for i in range(self.batch_size):
            self.inputs[i] = inputs[i]

        m_inputs = torch.from_numpy(self.inputs).view(
                self.batch_size, 3, 224, 224)
        if torch.cuda.is_available():
            m_inputs = m_inputs.cuda()

        return self.model(m_inputs)


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

    if args.model == "resnet18":
        underlying_model = torchvision.models.resnet18
    elif args.model == "resnet152":
        underlying_model = torchvision.models.resnet152

    model = ModelWrapper(underlying_model(), args.batch_size)

    if args.redundancy_mode == "cheap":
        red_model = ModelWrapper(MobileNetV2(width_mult=0.25), args.batch_size)
    else:
        red_model = RedModelWrapper(underlying_model(), args.batch_size)

    if args.task == "frontend":
        frontend(args)
    elif args.task == "worker":
        worker(args)
    else:
        assert False, "Unrecognized task '{}'".format(args.task)

