#!/usr/bin/env bash

: "${NPROCS:=16}"
: "${NTHREADS:=64}"
: "${BASE_OUTPUT_DIR:=/datasets/laion2B-en-aesthetic}"

while [ "$1" != "" ]; do
    case $1 in
        -j | --processes )      shift
                                NPROCS=$1
                                ;;
        -t | --threads  )       shift
                                NTHREADS=$1
                                ;;
        -o | --output-dir )     shift
                                BASE_OUTPUT_DIR=$1
                                ;;
    esac
    shift
done

URLS_DIR=${BASE_OUTPUT_DIR}/urls
WEBDATASET_DIR=${BASE_OUTPUT_DIR}/webdataset

for i in {00000..00127}; do wget -N -P ${URLS_DIR} https://huggingface.co/datasets/laion/laion2B-en-aesthetic/resolve/main/part-$i-9230b837-b1e0-4254-8b88-ed2976e9cee9-c000.snappy.parquet; done

img2dataset \
    --url_list urls \
    --input_format "parquet" \
    --url_col "URL" \
    --caption_col "TEXT" \
    --output_format webdataset \
    --output_folder ${WEBDATASET_DIR} \
    --processes_count ${NPROCS} \
    --thread_count ${NTHREADS} \
    --incremental_mode "incremental" \
    --resize_mode "no" \
    --save_additional_columns '["similarity","hash","punsafe","pwatermark","aesthetic"]' \
    --enable_wandb False

# TODO(ahmadki): the download process is non deterministic, we need a better solution for the final benchmark

