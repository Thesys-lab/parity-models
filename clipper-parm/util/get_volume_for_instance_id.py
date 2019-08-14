import argparse
import boto3
import pdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("instance_id", type=str, help="ID of instance to query")
    args = parser.parse_args()

    client = boto3.client('ec2')
    volumes = client.describe_volumes(Filters=[{
            "Name": "attachment.instance-id",
            "Values": [args.instance_id]
        }])
    volume_ids = [v["VolumeId"] for v in volumes["Volumes"]]
    print(volume_ids)
