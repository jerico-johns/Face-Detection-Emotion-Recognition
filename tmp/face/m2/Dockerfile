FROM nvcr.io/nvidia/pytorch:21.05-py3

# RUN pip3 install papermill

VOLUME /notebooks
WORKDIR /notebooks
EXPOSE 8888

CMD jupyter notebook --no-browser --ip=0.0.0.0 --allow-root
