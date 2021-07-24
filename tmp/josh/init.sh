docker build -t proj .
docker run --gpus all -it --rm -v $(pwd)/notebooks:/notebooks -p 8888:8888 --shm-size=4096m proj