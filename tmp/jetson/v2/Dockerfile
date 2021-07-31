ARG BASE_IMAGE=nvcr.io/nvidia/l4t-ml:r32.4.3-py3

FROM ${BASE_IMAGE} 

# RUN pip3 install papermill

RUN pip3 install --upgrade pip
RUN apt update
RUN apt install -y python3-dev python3-pip python3-opencv libopencv-dev
RUN pip3 install Cython
RUN pip3 install numpy
RUN pip3 install facenet-pytorch
# RUN pip3 install mmcv

VOLUME /proj_files
WORKDIR /proj_files
EXPOSE 8888

CMD jupyter notebook --no-browser --ip=0.0.0.0 --allow-root
