""" Utility script for launching EC2 instances """

import argparse
import boto3
import time


EXP_ID_TAG = "EXP_ID"
TASK_TYPE_TAG = "TASK_TYPE"

cpu_ami = "ami-0a6dbf5c079273797"
client_ami = "ami-094210d8b9070c6b1"
gpu_ami = "ami-0d48daa1974a90f5e"

security_groups = ["ParmSecurityGroup"]

target_availability_zone = "us-west-2a"


def launch_instances(args):
    ec2 = boto3.resource('ec2')
    # Get list of instances that have our
    # exp_id tag and designated instance tpe
    available_instances = list(ec2.instances.filter(Filters=[
        { "Name": "instance-type", "Values": [args.instance_type] },
        { "Name": "availability-zone", "Values": [target_availability_zone] },
        { "Name": "tag:{}".format(EXP_ID_TAG), "Values": [args.exp_tag] },
        { "Name": "tag:{}".format(TASK_TYPE_TAG), "Values": [args.task_tag] }
    ]))

    if len(available_instances) > 0 and not args.reuse:
        raise Exception(("Found {} instances, but have not specified to --reuse"
                         "Only one set of running instances can have EXP_ID").format(len(running_instances)))

    running_instances = [i for i in available_instances if i.state['Name'] == "running"]

    num_instances = args.num_instances

    if len(running_instances) < num_instances:
        stopped_instances = [i for i in available_instances if i.state['Name'] == "stopped"]

        num_to_start = min(len(stopped_instances), num_instances - len(running_instances))
        started_instances = []
        for instance in stopped_instances[:num_to_start]:
            instance.start()
            started_instances.append(instance)

        num_to_launch = num_instances - len(running_instances) - num_to_start

        if args.task_tag in ["client", "bg_client", "background"]:
            image_id = client_ami
        elif args.task_tag == "worker" and "p2" in args.instance_type:
            image_id = gpu_ami
        else:
            image_id = cpu_ami

        if num_to_launch > 0:
            # TODO: Add support for launching spot instances
            print("Launching", num_to_launch, "instances")
            placement_dict = {"AvailabilityZone": target_availability_zone}
            launched_instances = ec2.create_instances(
                ImageId=image_id,
                SecurityGroups=security_groups,
                InstanceType=args.instance_type,
                MinCount=num_to_launch,
                MaxCount=num_to_launch,
                Placement=placement_dict
            )

            instance_ids = [i.id for i in launched_instances]
            response = ec2.create_tags(
                    Resources=instance_ids,
                    Tags=[
                        {
                            'Key': 'Name',
                            'Value': "clipper-" + args.task_tag
                            },
                        {
                            'Key': EXP_ID_TAG,
                            'Value': args.exp_tag
                            },
                        {
                            'Key': TASK_TYPE_TAG,
                            'Value': args.task_tag
                            }
                        ])
            started_instances.extend(launched_instances)

        for instance in started_instances:
            instance.wait_until_running()
            while instance.public_ip_address is None:
                instance.reload()
                time.sleep(2)

        print("All instances launched!")
        instances = running_instances + started_instances
    else:
        print("Using the first", num_instances, "available instances")
        instances = running_instances[:num_instances]

    instances = sorted(instances, key=lambda x: x.public_ip_address)

    # Save all launched private ips, but only the first |args.num_instances| public ips
    all_instance_ips = [i.public_ip_address for i in instances]
    all_instance_ids = [i.instance_id for i in instances]

    w_inst = args.num_instances
    if args.mode == "none" and args.task_tag == "worker":
        w_inst = args.num_instances - args.red_models
    instance_ips = all_instance_ips[:w_inst]
    instance_private_ips = [i.private_ip_address for i in instances]
    print(len(instance_ips), "IPS")
    print(instance_ips)
    print(len(instance_private_ips), "PRIVATE IPS")
    print(instance_private_ips)

    to_save_file = args.save_ip_file
    with open(to_save_file, 'w') as outfile:
        outfile.write(' '.join(instance_ips))
    print("Saved ip addresses to", to_save_file)

    if args.task_tag == "worker":
        to_save_file = args.save_ip_file.split('.txt')[0] + "_all.txt"
        with open(to_save_file, 'w') as outfile:
            outfile.write(' '.join(all_instance_ips))
        print("Saved total ip addresses to", to_save_file)

    if args.save_private_ip_file:
        to_save_file = args.save_private_ip_file
        with open(to_save_file, 'w') as outfile:
            outfile.write(' '.join(instance_private_ips))
        print("Saved private ip addresses to", to_save_file)

    if args.save_instance_ids_file:
        to_save_file = args.save_instance_ids_file
        with open(to_save_file, 'w') as outfile:
            outfile.write(' '.join(all_instance_ids))
        print("Saved instance ids to", to_save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_tag", type=str,
                        help="Id to tag resources with")
    parser.add_argument("task_tag", type=str,
                        choices=["frontend", "worker", "client",
                                 "background", "bg_frontend", "bg_client"],
                        help="Task of instances to launch.")
    parser.add_argument("num_instances", type=int,
                        help="Number of instances to launch")
    parser.add_argument("mode", type=str,
                        choices=["none", "equal", "coded", "cheap"],
                        help="One of {none, none_single_queue, coded}")
    parser.add_argument("--instance_type", type=str,
                        help="Type of instance to launch", default="t2.micro")
    parser.add_argument("--reuse", action="store_true",
                        help="Whether to reuse existing resources that have tag exp_tag")
    parser.add_argument("--save_ip_file", type=str,
                        help="Location to save ip addresses to")
    parser.add_argument("--save_private_ip_file", type=str,
                        help="Location to save ip addresses to")
    parser.add_argument("--save_instance_ids_file", type=str,
                        help="Location to save instance ids to")
    parser.add_argument("--red_models", type=int,
                        help="Number of redundant instances", default=0)
    args = parser.parse_args()
    print(args)

    launch_instances(args)
