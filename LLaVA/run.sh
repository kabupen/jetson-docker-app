#/bin/bash

docker container run -it --rm  \
    -v /home/nvidia/.cache/huggingface/:/root/.cache/huggingface \
    jetson/llava bash

# python3 -m llava.serve.cli \
#     --model-path liuhaotian/llava-v1.5-7b \
#     --image-file "https://llava-vl.github.io/static/images/view.jpg"