#!/usr/bin/env bash

: "${SRC_IMG:=gitlab-master.nvidia.com/dl/dgx/pytorch:22.12-py3-devel}"
: "${DST_IMG:=gitlab-master.nvidia.com/akiswani/images/mlperf_sd_reference:22.12-py3-devel}"

while [ "$1" != "" ]; do
    case $1 in
        -s | --src-img )        shift
                                SRC_IMG=$1
                                ;;
        -d | --dst-img  )       shift
                                DST_IMG=$1
                                ;;
    esac
    shift
done

docker build -f docker/Dockerfile . --rm -t ${DST_IMG} --build-arg FROM_IMAGE_NAME=${SRC_IMG}
