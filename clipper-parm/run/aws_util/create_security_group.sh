#!/bin/bash

tmpfile=/tmp/parm_secgroup
group_name=ParmSecurityGroup
aws ec2 create-security-group --group-name $group_name \
                              --description "Security group for parity models" > $tmpfile

if [ "$?" -ne "0" ]; then
  echo "Error creating security group"
  rm $tmpfile
  exit 1
fi

if [ ! -d $PARM_LOCAL_HOME/aws_config ]; then
  mkdir $PARM_LOCAL_HOME/aws_config
fi

security_group_file=$PARM_LOCAL_HOME/aws_config/security_group.txt
tail -n 2 $tmpfile | head -n 1 | awk -F'"' '{print $4}' > $security_group_file
rm $tmpfile

# Create communication rules for security group


# Allow SSH
aws ec2 authorize-security-group-ingress --group-name $group_name \
                                        --protocol tcp --port 22 \
                                        --cidr 0.0.0.0/0

# Allow all communication on ports 7000 and 1337
aws ec2 authorize-security-group-ingress --group-name $group_name \
                                        --protocol tcp --port 7000 \
                                        --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-name $group_name \
                                        --protocol tcp --port 1337 \
                                        --cidr 0.0.0.0/0

# Allow all communication from others in our security group
aws ec2 authorize-security-group-ingress --group-name $group_name \
                                        --protocol tcp --port 1-65535 \
                                        --source-group $group_name
aws ec2 authorize-security-group-ingress --group-name $group_name \
                                        --protocol icmp --port -1 \
                                        --source-group $group_name

