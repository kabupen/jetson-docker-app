#!/bin/bash

docker container run -it --rm -v `pwd`:/work -w /work jda:yolo bash
