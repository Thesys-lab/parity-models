# Latency experiments
This portion of the repository contains code used for evaluating the tail
latency obtained when using parity models in a prediction serving system.

The experiments found in the paper are run on AWS with a large number of
instances (e.g., 20-40). This repository includes scripts for launching a
cluster of instances and performing these experiments. The VM images used
in AWS are made public as: "parity-models-cpu", "parity-models-gpu", and
"parity-models-client".

We first describe how to run these experiments in a distributed system on AWS,
which enables reproducing the experiments in the paper. We then describe how
one can run a simpler example on a single machine that executes many of the
same scripts as the distributed version (this single-machine version will not
provide meaningful results).

The code required for running each of these settings is found in the [run](run)
directory.

## Preliminaries
For both distributed and single-machine scripts to execute properly, the
following commands must be executed:
```bash
export PARM_LOCAL_HOME=/path/to/parity-models/clipper-parm
```
Consider adding this line to your `.bashrc` or `.bash_profile` if you plan
to experiment with this repository for an extended period of time.

## Running experiments on AWS
Running experiments on AWS requires additional setup:
1. Set up the [AWS CLI](https://docs.aws.amazon.com/polly/latest/dg/setup-aws-cli.html), if you have not already.
2. Set the following environment variables:
```bash
export PARM_SSH_KEY=/path/to/my/aws/ssh/key.pem
```
3. Create an AWS security group that will be used by instances in experiments.
```bash
cd run
./aws_util/create_security_group.sh
```

Your environment should now be prepared for launching and running experiments
on AWS.

The configuration files used for all experiments are found in the
[run/config](run/config) directory. Each JSON configuration file contains
experiment parameters used for generating one figure or primary result in the
paper. Each of these scripts may be executed in the following manner (here,
using as an example the script for generating Figure 9a):
```bash
cd run
python3 run_exp.py config/fig9a.json launch 
```

This will perform a number of actions:
1. Launch AWS instances, if not already created
2. Rsync the current `../src` to the frontend node and `../clipper_admin` to
all nodes. This will take a long time the first time this script is called, but
should be much quicker for subsequent experiments, unless you make substantial
changes to either of these directories locally.
3. Compile `clipper-parm` on the frontend and build worker Docker images on
each worker instance. Docker images are rebuilt on each invocation in case
contents of `../clipper_admin` change. This can likely be avoided if necessary,
but the overhead of rebuilding the images is currently not large enough to
merit the change.
4. Start model instances on workers and frontend-related components on the
frontend. "Frontend-related components" include the Clipper frontend, Redis,
etc. Each component exists in a separate Docker container. Containers can be
queried by running `docker ps` on the frontend; those related to the frontend
will contain the postfix `-parm`.
5. Run experiments determined by the configuration file. Please see 
[documentation](run/config/README.md) for an explanation of available
configuration parameters.
6. Save results on the remote frontend node in locations printed out at the
end of the script. These are typically located in `/tmp` on the host and have
a final file name of `exp_files.tar.gz`.

The first 3 steps above involve a connection between the computer from which
you are launching these experiments and the various nodes involved in the
experiment. These steps are complete when the message `Starting run on frontend`
appears. From steps 4 on, this connection is no longer necessary, and
experiments will continue to execute even if the ssh connection terminates.
In this event, the final destinations to which experimental results are saved
will be printed at the bottom of `~/runne_log.txt` on the frontend instance.

If your ssh connection remains active through the duration of the experiments,
the local runner will copy back results from the remote frontend to your local
machine and print stats. If this connection fails, you may retrieve stats after
the fact using:
```bash
./get_results.sh remote/path/to/results 0
```
where `remote/path/to/results` is the path printed out by your local script
(e.g., `ParM/wt_c5.xlarge/bm_resnet18/bg_0/bs_1/rate_4/k_2/nq_304/mode_equal/it_0/0`).

For the `config/simple.json` configuration file, this should result in results
being printed that look something like:
```
E2E_LATENCY (ms)
NUM   MED   MEAN  P99     P99.5   P99.9
100   80.16 85.92 250.56  258.91  265.59
```

Encoding/decoding statistics are only printed when using the `coded`
redundancy mode.

### Terminating AWS instances
When you have finished running experiments, you may terminate AWS instances
using:
```bash
cd run
./aws_util/terminate_instances.sh
```

This will print JSON describing the instances that whose state has been changed
to "shutting-down":
```
{
    "TerminatingInstances": [
        {
            "InstanceId": "i-09a36cb4ba2396d91", 
            "CurrentState": {
                "Code": 32, 
                "Name": "shutting-down"
            }, 
            "PreviousState": {
                "Code": 16, 
                "Name": "running"
            }
        }, 
        ...
]
```

### Don't want to run the large experiments, but want to get a distributed setup working?
Try using our [example](run/config/simple.json) configuration that only requires
3 worker instances, one frontend instance, and one client instance. This
configuration has not been set up to provide meaningful results, and thus should
not be relied upon for reproducing results.

## Single-machine setup
If you would like to test out getting ParM set up, but don't want to deal with
setting up the distributed system, consider trying out running on a single
machine. We have experimented with this setup using the AWS Ubuntu 16.04 x86 AMI,
but we expect that any Ubuntu 16.04 (e.g., one on CloudLab) should suffice. The
setup script will require around 30 GB of disk space. The setup assumes that your
home directory is `/home/ubuntu`, as is the case on the AWS image.

To set up a machine, run:
```bash
./setup.sh
```
Note that you will need to log out and log back in to your machine, and then
execute a followup command (see below) for Docker non-root permissions to
take effect. You will be prompted to log out and log back into your machine,
after which you should execute:
```bash
./setup.sh continue
```

You can then run local experiments using:
```bash
cd run
PARM_LOCAL_HOME=/home/ubuntu/parity-models/clipper-parm python3 run_exp.py config/simple.json launch --local
```

This will proceed as described above for distributed experiments, but all
frontend, model instance, and client containers and processes will be run
on the same node.

**NOTE:** Depending on the capabilities of your machine, this may run slowly
or overwhelm the machine. We have therefore developed "dummy" models
to be used specifically in this setting in [local_exp.py](run/local_exp.py). The
models deployed here do not actually perform inference, they just return a
list of floats. Models are thus computationally inexpensive and unlikely to
overwhelm a system. We note that this dummy variant is **NOT** intended to
be used for performance evaluation, but rather for exercising the scripts
needed to launch an experiment.
