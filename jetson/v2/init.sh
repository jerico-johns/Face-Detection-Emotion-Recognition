export DISPLAY=:0
sudo xhost +
docker build -t proj .
docker run --gpus all -it --rm -v $(pwd)/../..:/proj_files -p 8888:8888 --shm-size=4096m --privileged --device /dev/video0 -e DISPLAY=$DISPLAY --runtime nvidia proj
