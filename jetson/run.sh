#!/bin/bash
sudo docker run --privileged  -it --rm --device /dev/video0 -e DISPLAY=$DISPLAY --runtime nvidia --network host -v ~/mids-w251-project:/mids-w251-project mids-w251-project:1 bash
