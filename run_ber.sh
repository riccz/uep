#!/bin/bash

set -eu -o pipefail
set -x

BUILD=FAILED
PREPARE_VIDEO=FAILED
RUN=FAILED
PROCESS_OUTPUT=FAILED

UPLOADED_FILES=""

function send_aws_mail {
    tmpfile=$(mktemp)
    echo "$1" >> "$tmpfile"
    echo "BUILD = ${BUILD}" >> "$tmpfile"
    echo "PREPARE_VIDEO = ${PREPARE_VIDEO}" >> "$tmpfile"
    echo "RUN = ${RUN}" >> "$tmpfile"
    echo "PROCESS_OUTPUT = ${PROCESS_OUTPUT}" >> "$tmpfile"
    echo "--- UPLOADED_FILES ---" >> "$tmpfile"
    echo -e "${UPLOADED_FILES}" >> "$tmpfile"

    aws sns publish \
	--region 'us-east-1' \
	--topic-arn 'arn:aws:sns:us-east-1:402432167722:NotifyMe' \
	--message "file://${tmpfile}"
    rm "${tmpfile}"
}

trap "send_aws_mail 'There was an error'" ERR

# Get dependencies
sudo apt-get update
sudo apt-get install -y \
     build-essential \
     cmake \
     git \
     libboost-all-dev \
     libprotobuf-dev \
     libpthread-stubs0-dev \
     p7zip \
     protobuf-compiler

# Clone repo
git clone --recursive https://github.com/riccz/uep.git

# Build
cd uep
git checkout master
mkdir -p build
pushd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
popd

BUILD=OK

# Prepare video
pushd dataset
wget 'http://trace.eas.asu.edu/yuv/stefan/stefan_cif.7z'
7zr x stefan_cif.7z
popd
pushd h264_scripts
./encode.sh
popd

PREPARE_VIDEO=OK

# Run
pushd build/bin
./server -K '[50, 950]' -R '[1, 1]' -E 1 -n 1080 -r 1000000 > server_console.log 2>&1 &
sleep 10
./client -n stefan_cif_long -t 30 -p 0 > client_console.log 2>&1 &
wait

# Upload logs
subdir_name=$(date +'%Y-%m-%d_%H-%M-%S')
for l in *.log; do
    s3_url="s3://uep.zanol.eu/simulation_logs/${subdir_name}/${l}"
    aws s3 cp "$l" "$s3_url"
    publink=$(aws s3 presign "$s3_url" --expires-in 31536000)
    UPLOADED_FILES="${UPLOADED_FILES}${l}: ${publink}\n"
done
popd

RUN=OK

# Process output video
pushd h264_scripts
./decode.sh
popd

# Upload videos
pushd dataset_client
aws s3 cp "stefan_cif_long.264" s3://uep.zanol.eu/simulation_videos/"$subdir_name"/"stefan_cif_long.264"
popd

PROCESS_OUTPUT=OK

send_aws_mail "run_ber is finished ($subdir_name)"

# Shutdown
sudo poweroff
