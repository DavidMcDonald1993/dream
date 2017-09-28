FROM ubuntu:16.04
MAINTAINER David McDonald "dxm237@cs.bham.ac.uk"

## dependences
RUN apt-get update
RUN apt-get -y install python python-pip
RUN apt-get -y install libhdf5-dev

# make data and output directories
RUN mkdir /evaluation_data
RUN mkdir /output

## python packages
RUN pip install --upgrade pip
RUN pip install numpy 
RUN pip install pandas  
RUN pip install fancyimpute 
RUN pip install keras
RUN pip install h5py

# add files
ADD subchallenge_2_submission.py /subchallenge_2.py
ADD model.json /model.json
ADD model_weights.h5 /model_weights.h5
ADD pca_scaler.pkl /pca_scaler.pkl
ADD score_sc2.sh /score_sc2.sh

CMD [ "./score_sc2.sh" ]