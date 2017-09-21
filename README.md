# dream

The results directory contains rmses for subchallenges 1 and 2 respectively. 

In subchallenge 1: 
"no_mask" is the rmse between the ground truth and the predictions, including where the ground truth is zero.
"ignore_zeros" is the rsme between ground truth and predictions, ignoring where the ground truth is 0 (I believe this is a missing value across all 10 datasets)
"only_nan_ignore_zeros" only computes rmse between the predictions and ground truth where the original data is missing, but the gorund truth has the correct value


The google spreadsheet with the results is here:
https://docs.google.com/spreadsheets/d/1SUJ0gBnSRGwjWPW5TwKU4Wbldgkn6Mod7LvB6V4FTPw/edit#gid=1286755451
