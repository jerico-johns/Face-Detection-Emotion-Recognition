#!/bin/bash
docker run --rm --gpus all --ipc=host --net=host -it --shm-size=2048m --name jupyter -p 8888:8888 -v /data:/data -v /home/ubuntu/mids-w251-project:/mids-w251-project mids-w251-project-aws:1 bash
