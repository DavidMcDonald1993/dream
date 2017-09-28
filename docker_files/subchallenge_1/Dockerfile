FROM ubuntu:16.04
MAINTAINER David McDonald "dxm237@cs.bham.ac.uk"

## dependences
RUN apt-get update
RUN apt-get -y install python python-pip

# make data and output directories
RUN mkdir /evaluation_data
RUN mkdir /output

## python packages
RUN pip install --upgrade pip
RUN pip install numpy 
RUN pip install pandas  
RUN pip install fancyimpute 

# add files
ADD subchallenge_1_submission.py /subchallenge_1.py
ADD score_sc1.sh /score_sc1.sh

CMD [ "./score_sc1.sh" ]