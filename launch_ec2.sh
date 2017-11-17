#!/bin/bash

set -eu -o pipefail

if [ "$#" -eq 3 ]; then
    ec2type="${1}"
    size="${2}"
    script="${3}"
elif [ "$#" -eq 2 ]; then
    ec2type="${1}"
    size=8
    script="${2}"
elif [ "$#" -eq 1 ]; then
    ec2type="c5.large"
    size=8
    script="${1}"
else
    ec2type="c5.large"
    size=8
    script=""
fi

instance_id=$(aws ec2 run-instances \
                  --region us-east-1 \
                  --image-id ami-5cbe7326 \
                  --block-device-mappings "DeviceName=xvda,Ebs={DeleteOnTermination=true,VolumeSize=${size},VolumeType=gp2}" \
                  --count 1 \
                  --instance-type "${ec2type}" \
                  --subnet-id 'subnet-0c601120' \
                  --security-group-ids 'sg-d030c9a0' \
                  --key-name 'gpg auth key' \
                  --iam-instance-profile 'Name=uep_sim_server' \
                  --instance-initiated-shutdown-behavior 'terminate' \
                  --associate-public-ip-address | \
                  grep 'InstanceId' | sed 's/.*: "\(.*\)".*/\1/')

while true; do
    set +e +o pipefail
    descr_out=$(aws ec2 describe-instances \
                 --region us-east-1 \
                 --instance-ids "$instance_id")

    ipaddr=$(grep 'PublicIpAddress' <<< "$descr_out" | \
                 sed 's/.*: "\(.*\)".*/\1/')
    set -e -o pipefail

    if [ -n "$ipaddr" ]; then
        break
    fi
    sleep 10
done

while true; do
    set +e +o pipefail
    ssh -o ConnectTimeout=10 \
        -o UserKnownHostsFile=/dev/null \
        -o StrictHostKeyChecking=no \
        admin@"$ipaddr" <<< "echo SSH OK"
    ssh_rc="$?"
    set -e -o pipefail

    if [ "$ssh_rc" == "0" ]; then
        break
    fi
    sleep 10
done

ssh -o ConnectTimeout=10 \
    -o UserKnownHostsFile=/dev/null \
    -o StrictHostKeyChecking=no \
    admin@"$ipaddr" < setup_script.sh

if [ -z "$script" ]; then
    # Open interactive SSH
    ssh -o ConnectTimeout=10 \
        -o UserKnownHostsFile=/dev/null \
        -o StrictHostKeyChecking=no \
        admin@"$ipaddr"
else
    # Send the script
    script_base64=$(base64 "$script")
    ssh -o ConnectTimeout=10 \
        -o UserKnownHostsFile=/dev/null \
        -o StrictHostKeyChecking=no \
        -t admin@"$ipaddr" <<EOF
          set -eu -o pipefail
          base64 -d <<< "${script_base64}" > "${script}"
          chmod a+x "${script}"
          screen -d -m -L screen.log python3 "${script}"
          set +eu +o pipefail
          screen -r
EOF
fi

# Local Variables:
# indent-tabs-mode: nil
# End:
