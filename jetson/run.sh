#!/bin/bash
export DISPLAY=:0
sudo xhost +
sudo docker run --gpus all --privileged  -it --rm --device /dev/video0 -e DISPLAY=$DISPLAY --runtime nvidia --network host -v ~/mids-w251-project:/mids-w251-project mids-w251-project:2 bash
