# Using ubuntu 18.04 as base image
#FROM nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3 
FROM nvcr.io/nvidia/l4t-pytorch:r32.4.4-pth1.6-py3
# update the base image
RUN apt-get update && apt-get -y update
# install face_recognition
RUN apt-get -y install python3-pip cmake libopenblas-dev liblapack-dev libjpeg-dev
RUN pip3 install numpy
WORKDIR /mids-w251-project
RUN wget http://dlib.net/files/dlib-19.17.tar.bz2 
RUN tar jxvf dlib-19.17.tar.bz2
WORKDIR /mids-w251-project/dlib-19.17
RUN ls -al
RUN sed -i 854d dlib/cuda/cudnn_dlibapi.cpp
RUN python3 setup.py install
RUN pip3 install face_recognition
# install emotion_recognition prerequisites
RUN apt -y install python3-opencv
RUN pip3 install matplotlib
RUN apt -y install vim


#RUN pip3 install --upgrade pip
#RUN apt update
RUN apt install -y python3-dev python3-pip python3-opencv libopencv-dev
RUN pip3 install Cython
#RUN pip3 install numpy
RUN pip3 install facenet-pytorch
# RUN pip3 install mmcv

# Added Jerico's libraries
#RUN pip3 install mmcv
#RUN pip3 install MTCNN

# Optional for jupyter install

RUN apt-get install python3-pip python3-dev build-essential nodejs -y

# make python3 -> python
RUN ln -s /usr/bin/python3 /usr/local/bin/python 
# update pip
RUN pip3 install --upgrade pip
# install jupyter and lab
RUN pip3 install jupyter
RUN pip3 install jupyterlab
# set our workdir
WORKDIR /mids-w251-project
#COPY notebooks/emotion.ipynb ./
# Setup which command to run...
# This runs jup notebook 
#CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
# This runs jup lab
#CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
