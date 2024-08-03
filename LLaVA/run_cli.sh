#!/bin/bash

docker container run -it --rm \
    -v /home/nvidia/.cache/huggingface/:/root/.cache/huggingface \
    -v /home/nvidia/work/jetson-docker-app/LLaVA:/tmpdata \
    -v /data:/data \
    jetson/llava \
    bash