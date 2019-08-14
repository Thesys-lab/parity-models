""" A single worker model instance """
import argparse
import docker
import logging
import os
import random
import socket
import struct
import sys
import tarfile
import tempfile
import threading
import time

logger = logging.getLogger(__name__)

if sys.version_info < (3, 0):
    try:
        from cStringIO import StringIO
    except ImportError:
        from StringIO import StringIO
    PY3 = False
else:
    from io import BytesIO as StringIO
    PY3 = True


from ..container_manager import (
    create_model_container_label, parse_model_container_label,
    ContainerManager, CLIPPER_DOCKER_LABEL, CLIPPER_MODEL_CONTAINER_LABEL,
    CLIPPER_QUERY_FRONTEND_CONTAINER_LABEL,
    CLIPPER_MGMT_FRONTEND_CONTAINER_LABEL, CLIPPER_INTERNAL_RPC_PORT,
    CLIPPER_INTERNAL_QUERY_PORT, CLIPPER_INTERNAL_MANAGEMENT_PORT,
    CLIPPER_INTERNAL_METRIC_PORT)

from .common import *


class ModelInstanceWorker:
    def __init__(self, name, input_type, func, redundant,
                 query_frontend_hostname, serialization_fn,
                 version=1, pkgs_to_install=None, location="/home/ubuntu/clipper-parm",
                 model=None, gpu=True):
        self.name = name
        self.version = version
        self.func = func
        self.input_type = input_type
        self.redundant = redundant
        self.query_frontend_hostname = query_frontend_hostname
        self.extra_container_kwargs = {}
        self.serialization_fn = serialization_fn
        self.pkgs_to_install = pkgs_to_install
        self.common_labels = {CLIPPER_DOCKER_LABEL: ""}
        self.location = location
        if os.path.isdir("/home/ubuntu/parity-models/clipper-parm"):
            self.location = "/home/ubuntu/parity-models/clipper-parm"
        self.model = model
        self.gpu = gpu

    def build_image(self):
        if not self.model:
            model_data_path, base_image = self.serialization_fn(self.name, self.func)
        else:
            model_data_path, base_image = self.serialization_fn(self.name, self.func, self.model)

        version = str(self.version)

        #_validate_versioned_model_name(name, version)

        run_cmd = ''
        if self.pkgs_to_install:
            run_as_lst = 'RUN apt-get -y install build-essential && pip install'.split(
                ' ')
            run_cmd = ' '.join(run_as_lst + self.pkgs_to_install)
        with tempfile.NamedTemporaryFile(
                mode="w+b", suffix="tar") as context_file:
            # Create build context tarfile
            with tarfile.TarFile(
                    fileobj=context_file, mode="w") as context_tar:
                context_tar.add(model_data_path)
                # From https://stackoverflow.com/a/740854/814642
                try:
                    df_contents = StringIO(
                        str.encode(
                            "FROM {container_name}\nCOPY {data_path} /model/\n{run_command}\n".
                            format(
                                container_name=base_image,
                                data_path=model_data_path,
                                run_command=run_cmd)))
                    df_tarinfo = tarfile.TarInfo('Dockerfile')
                    df_contents.seek(0, os.SEEK_END)
                    df_tarinfo.size = df_contents.tell()
                    df_contents.seek(0)
                    context_tar.addfile(df_tarinfo, df_contents)
                except TypeError:
                    df_contents = StringIO(
                        "FROM {container_name}\nCOPY {data_path} /model/\n{run_command}\n".
                        format(
                            container_name=base_image,
                            data_path=model_data_path,
                            run_command=run_cmd))
                    df_tarinfo = tarfile.TarInfo('Dockerfile')
                    df_contents.seek(0, os.SEEK_END)
                    df_tarinfo.size = df_contents.tell()
                    df_contents.seek(0)
                    context_tar.addfile(df_tarinfo, df_contents)
            # Exit Tarfile context manager to finish the tar file
            # Seek back to beginning of file for reading
            context_file.seek(0)
            image = "{name}:{version}".format(name=self.name, version=version)
            """if container_registry is not None:
                image = "{reg}/{image}".format(
                    reg=container_registry, image=image)"""
            self.docker_client = docker.from_env()
            logger.info(
                "Building model Docker image with model data from {}".format(
                    model_data_path))
            image_result, build_logs = self.docker_client.images.build(
                fileobj=context_file, custom_context=True, tag=image)
            for b in build_logs:
                logger.info(b)

        logger.info("Pushing model Docker image to {}".format(image))
        for line in self.docker_client.images.push(repository=image, stream=True):
            logger.debug(line)
        return image

    def add_model(self):
        self.image = self.build_image()
        print("Built and pushed image")
        env_vars = {
            "CLIPPER_MODEL_NAME": self.name,
            "CLIPPER_MODEL_VERSION": self.version,
            # NOTE: assumes this container being launched on same machine
            # in same docker network as the query frontend
            "CLIPPER_IP": self.query_frontend_hostname,
            "CLIPPER_INPUT_TYPE": self.input_type,
        }
        model_container_label = create_model_container_label(self.name, self.version, self.redundant)
        labels = self.common_labels.copy()
        labels[CLIPPER_MODEL_CONTAINER_LABEL] = model_container_label

        model_container_name = model_container_label + '-{}'.format(
            random.randint(0, 100000))

        volumes = {"{}/containers".format(self.location): {'bind': '/my_containers', 'mode': 'rw'}}
        if self.model:
            cmd = "/my_containers/python/container_entry.sh pytorch-container /my_containers/python/pytorch_container.py"
            if self.gpu:
                self.extra_container_kwargs["runtime"] = "nvidia"
        else:
            cmd = "/my_containers/python/container_entry.sh py-closure-container /my_containers/python/python_closure_container.py"
        print("Starting container")
        self.docker_client.containers.run(
            self.image,
            command=cmd,
            name=model_container_name,
            environment=env_vars,
            labels=labels,
            volumes=volumes,
            detach=True,
            network_mode="host",
            **self.extra_container_kwargs)
        print("Started container")
        return model_container_name

    def start(self):
        self.container_name = self.add_model()
        container = self.docker_client.containers.get(self.container_name)
        while container.attrs.get("State").get("Status") != "running" or \
                        self.docker_client.api.inspect_container(self.container_name).get("State").get("Health").get("Status") != "healthy":
            time.sleep(3)

    def stop(self):
        container = self.docker_client.containers.get(self.container_name)
        container.stop()

    def run(self, port):
        ip = "0.0.0.0"
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((ip, port))
        sock.listen(5)

        while True:
            (clientsocket, address) = sock.accept()
            print("Got connection")
            data = clientsocket.recv(msg_packer.size)
            msg = msg_packer.unpack(data)[0]

            print("msg is", msg)
            if msg == MSG_START:
                print("Got MSG_START")
                self.start()
                clientsocket.send(msg_packer.pack(*(msg,)))
                clientsocket.close()
                print("Responded to MSG_START")
            elif msg == MSG_STOP:
                print("Got MSG_STOP")
                self.stop()
                clientsocket.send(msg_packer.pack(*(msg,)))
                clientsocket.close()
                sock.close()
                print("Responded to MSG_STOP")
                break
