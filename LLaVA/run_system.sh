#!/bin/bash

docker container run -it --rm \
    -v /home/nvidia/.cache/huggingface/:/root/.cache/huggingface \
    -p 7860:7860 \
    -p 40000:40000 \
    -p 10000:10000 \
    jetson/llava \
    bash -c "\
        python -m llava.serve.controller --host 0.0.0.0 --port 10000& \
        python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload&
        python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1.5-7b&
        wait
        "
# python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1.5-7b&