FROM python:2

ADD subchallenge_1.py data_obs_1.txt data_obs_2.txt data_obs_3.txt data_obs_4.txt data_obs_5.txt data_obs_6.txt data_obs_7.txt data_obs_8.txt data_obs_9.txt data_obs_10.txt data_true.txt ./

RUN pip install numpy 
RUN pip install pandas 
RUN pip install scikit-learn 
RUN pip install fancyimpute 
RUN pip install tensorflow 
RUN pip install keras

CMD [ "python", "./subchallenge_1.py" ]