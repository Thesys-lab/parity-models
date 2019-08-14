# Running experiments
This subdirectory contains scripts used for running experiments in a
distributed setting. Distributed experiments currently support AWS only.
We provide public AMIs containing all software necessary for running
experiments. If you would like to attempt running on your own cluster, please
see [Running on your own cluster](#Running-on-your-own-cluster).

## Directory structure
* [config](config): JSON files containing configuration settings for
experiments in the paper as well as for running a small distributed test.

## Quickstart

### General setup (only ever need to execute once)
1. Set up the [AWS CLI](https://docs.aws.amazon.com/polly/latest/dg/setup-aws-cli.html), if you have not already.
2. Set the following environment variables:
```bash
export PARM_SSH_KEY=/path/to/my/aws/ssh/key.pem
export PARM_LOCAL_HOME=/path/to/parity-models/clipper-parm
```
3. Create an AWS security group that will be used by instances in experiments.
```bash
./aws_util/create_security_group.sh
```

### Launching AWS instances and running an experiment
1. *TODO*

### Tearing down AWS instances
You may terminate AWS instances that were launched above by using the following
command:
```bash
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
        {
            "InstanceId": "i-045e82a979adb2a86", 
            "CurrentState": {
                "Code": 32, 
                "Name": "shutting-down"
            }, 
            "PreviousState": {
                "Code": 16, 
                "Name": "running"
            }
        }
]
```

## Running on your own cluster
*TODO*
