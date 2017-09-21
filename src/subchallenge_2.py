from functools import partial

from keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

from fancyimpute import KNN

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error as mse

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, BayesianRidge

from sklearn.ensemble import AdaBoostRegressor

import xgboost as xgb

def preprocess_data(holdout=0.2):
    
    # import data for breast cancer datasets
    cna_data_frame = pd.read_csv("../data/sub_challenge_2_3/breast_cancer_dataset/retrospective_breast_CNA_median_sort_common_gene_16884.txt", 
                    header=0, index_col=0, sep="\t")
    rna_data_frame = pd.read_csv("../data/sub_challenge_2_3/breast_cancer_dataset/retrospective_breast_rna_seq_sort_common_gene_15115.txt", 
                    header=0, index_col=0, sep="\t")
    proteome_data_frame = pd.read_csv("../data/sub_challenge_2_3/breast_cancer_dataset/retrospective_breast_proteome_filtered.txt", 
                    header=0, index_col=0, sep="\t")

    # convert rna data to numpy array to impute
    rna_data = rna_data_frame.values

    # use knn to fill in missing values in rna data
    knn = KNN(k=5, verbose=0)
    rna_data = knn.complete(rna_data)

    # proteome data also has missing values, but I am ignoring rows with missing values for now
    # proteome_data = knn.complete(proteome_data)

    # update rna data frame with imputed data
    rna_data_frame = pd.DataFrame(rna_data, index = rna_data_frame.index, columns = rna_data_frame.columns)

    # find common proteins and patients across all datasets
    proteins = proteome_data_frame.index.intersection(cna_data_frame.index).intersection(rna_data_frame.index)
    patients = proteome_data_frame.columns.intersection(cna_data_frame.columns).intersection(rna_data_frame.columns)

    # locate common data and return as numpy array
    cna_data = cna_data_frame.loc[proteins, patients].values
    rna_data = rna_data_frame.loc[proteins, patients].values
    proteome_data = proteome_data_frame.loc[proteins, patients].values

    # masks for proteins with complete data and with missing value
    complete_proteins = proteins[~(np.isnan(proteome_data).any(axis=1))]
    missing_value_proteins = proteins[np.isnan(proteome_data).any(axis=1)]

    # use complete data as training/validation data
    cna_data = cna_data_frame.loc[complete_proteins, patients].values
    rna_data = rna_data_frame.loc[complete_proteins, patients].values
    proteome_data = proteome_data_frame.loc[complete_proteins, patients].values

    # standard scaling
    cna_scaler = StandardScaler()
    rna_scaler = StandardScaler()
    proteome_scaler = StandardScaler()

    cna_data = cna_scaler.fit_transform(cna_data)
    rna_data = rna_scaler.fit_transform(rna_data)
    proteome_data = proteome_scaler.fit_transform(proteome_data)
    
    # mask of training/validation data
    mask = np.random.rand(cna_data.shape[0]) < holdout
    
    cna_training_data = cna_data[~mask]
    rna_training_data = rna_data[~mask]
    proteome_training_data = proteome_data[~mask]
    
    cna_validation_data = cna_data[mask]
    rna_validation_data = rna_data[mask]
    proteome_validation_data = proteome_data[mask]
    
    # return training data scalers for inverse scaling
    return cna_training_data, rna_training_data, proteome_training_data, cna_validation_data, rna_validation_data, proteome_validation_data, cna_scaler, rna_scaler, proteome_scaler

def rmse (y_pred, y_true): 
    # rsme prediction and ground truth
    return np.sqrt(mse(y_true, y_pred))

# compute mean rmse across a number of repreats
def mean_rmse(cna_training_data, rna_training_data, proteome_training_data, 
            cna_validation_data, rna_validation_data, proteome_validation_data,
              regression_method, num_repeats=1, **kwargs):
    
    abundance_predictions = [regression_method(cna_training_data, rna_training_data, proteome_training_data, 
                                               cna_validation_data, rna_validation_data, **kwargs) 
                             for i in range(num_repeats)]
    
    rmses = np.array([rmse(abundance_prediction, proteome_validation_data) 
                      for abundance_prediction in abundance_predictions])

    return rmses.mean(axis=0)

def linear_regression(cna_training_data, rna_training_data, proteome_training_data, 
                                         cna_validation_data, rna_validation_data,**kwargs):
    
    print "Building linear model"
    
    linear_model = LinearRegression()
    
    print "Appending training matrices for linear model"
    
    # append data for linear regression
    cna_rna_training_data_append = np.append(cna_training_data, rna_training_data, axis=1)
    
    print "Fitting linear model"
    
    linear_model.fit(cna_rna_training_data_append, proteome_training_data)
    
    # predictions
    cna_rna_validation_data_append = np.append(cna_validation_data, rna_validation_data, axis=1)
    
    abundance_predictions = linear_model.predict(cna_rna_validation_data_append)
    
    return abundance_predictions

# construct model
def build_deep_regression_model(num_samples, num_hidden, dropout, activation, reg):
    
    regulariser = l2(reg)

    cna = Input(shape=(num_samples,))
    rna = Input(shape=(num_samples,))

    y = Concatenate()([cna, rna])

    for h in num_hidden:
        y = Dense(h, activation=activation, kernel_regularizer=regulariser)(y)
        y = BatchNormalization()(y)
        y = Dropout(dropout)(y)

    y = Dense(num_samples, kernel_regularizer=regulariser)(y)

    deep_non_linear_regression_model = Model([cna, rna], y)
    deep_non_linear_regression_model.compile(optimizer="adam", loss="mse")
    
    return deep_non_linear_regression_model

def deep_non_linear_regression(cna_training_data, rna_training_data, proteome_training_data,
                               cna_validation_data, rna_validation_data,
                               num_hidden=[128], dropout=0.1, activation="relu", reg=1e-3, **kwargs):
    
    print "Building deep non linear regression model"
    
    # dimensionality of data
    num_proteins, num_samples = cna_training_data.shape
     
    # build model
    regression_model = build_deep_regression_model(num_samples, num_hidden, dropout, activation, reg)

    early_stopping = EarlyStopping(monitor="loss", patience=1000, min_delta=0, )
    
    print "Training deep non linear regression model"

    # train model
    regression_model.fit([cna_training_data, rna_training_data], proteome_training_data, 
                         verbose=0, epochs=10000, batch_size=128, shuffle=True, validation_split=0.0,
                         callbacks=[early_stopping])

    # protein abundance predictions
    abundance_predictions = regression_model.predict([cna_validation_data, rna_validation_data])
    
    return abundance_predictions

def xgboost(cna_training_data, rna_training_data, proteome_training_data, 
            cna_validation_data, rna_validation_data,**kwargs):
    
    print "Reshaping training matrices for xgboost"
    
    # reshape to columns because xgboost regression can only have single numbers as labels
    cna_training_data_shaped = cna_training_data.reshape(-1, 1)
    rna_training_data_shaped = rna_training_data.reshape(-1, 1)
    proteome_training_data_shaped = proteome_training_data.reshape(-1, 1)
    
    # append cna and rna data
    cna_rna_training_data_append = np.append(cna_training_data_shaped, rna_training_data_shaped, axis=1)
    
    # construct training data matrix
    dtrain = xgb.DMatrix(cna_rna_training_data_append, label=proteome_training_data_shaped)
    
    print "Training xgboost"
    
    # train using default parameters
    bst = xgb.train(dtrain=dtrain, **kwargs)
    
    # abundance predicitions 
    cna_validation_data_shaped = cna_validation_data.reshape(-1, 1)
    rna_validation_data_shaped = rna_validation_data.reshape(-1, 1)
    cna_rna_validation_data_append = np.append(cna_validation_data_shaped, rna_validation_data_shaped, axis=1)
    dval = xgb.DMatrix(cna_rna_validation_data_append, label=proteome_training_data_shaped)
    abundance_predictions = bst.predict(dval)
    
    return abundance_predictions.reshape(cna_validation_data.shape)

def adaboost(cna_training_data, rna_training_data, proteome_training_data, 
             cna_validation_data, rna_validation_data, base_regressor, **kwargs):
    
    if base_regressor == "linear":
        regressor = LinearRegression()
    elif base_regressor == "lasso":
        regressor = Lasso()
    elif base_regressor == "elastic_net":
        regressor = ElasticNet()
    elif base_regressor == "bayesian_ridge":
        regressor = BayesianRidge()
        
    # reshape to columns because xgboost regression can only have single numbers as labels
    cna_training_data_shaped = cna_training_data.reshape(-1, 1)
    rna_training_data_shaped = rna_training_data.reshape(-1, 1)
    proteome_training_data_shaped = proteome_training_data.reshape(-1, )

    # append cna and rna data
    cna_rna_training_data_append = np.append(cna_training_data_shaped, rna_training_data_shaped, axis=1)    
    
    # adabosst object
    adaboost = AdaBoostRegressor(base_estimator=regressor)
    
    print "fitting adaboost with {}".format(base_regressor)
    
    adaboost.fit(cna_rna_training_data_append, proteome_training_data_shaped)
    
    print "predicting"
    
    cna_validation_data_shaped = cna_validation_data.reshape(-1, 1)
    rna_validation_data_shaped = rna_validation_data.reshape(-1, 1)
    cna_rna_validation_data_append = np.append(cna_validation_data_shaped, rna_validation_data_shaped, axis=1)
    
    abundance_predictions = adaboost.predict(cna_rna_validation_data_append)
    
    return abundance_predictions.reshape(cna_validation_data.shape)

def main():
    
    print "Preprocessing data"

    # get training data
    cna_training_data, rna_training_data, proteome_training_data, \
    cna_validation_data, rna_validation_data, proteome_validation_data, \
    cna_scaler, rna_scaler, proteome_scaler = preprocess_data()
    
    # different deep architechitures
    deep_1 = partial(deep_non_linear_regression, activation="sigmoid", num_hidden=[256])
    deep_2 = partial(deep_non_linear_regression, activation="relu", num_hidden=[256])
    deep_3 = partial(deep_non_linear_regression, activation="tanh", num_hidden=[256])
    
    # different xgboost parameters
    xgboost_4 = partial(xgboost, params={"max_depth" : 4})
    xgboost_6 = partial(xgboost, params={"max_depth" : 6})
    xgboost_10 = partial(xgboost, params={"max_depth" : 10})
    xgboost_0 = partial(xgboost, params={"max_depth" : 0})
    
    # adaboost with different base estimators
    adaboost_linear = partial(adaboost, base_regressor="linear")
    adaboost_lasso = partial(adaboost, base_regressor="lasso")
    adaboost_elastic_net = partial(adaboost, base_regressor="elastic_net")
    adaboost_bayesian_ridge = partial(adaboost, base_regressor="bayesian_ridge")
    
    #list of regression methods
    regression_methods = [linear_regression, deep_1, deep_2, deep_3,
                          xgboost_4, xgboost_6, xgboost_10, xgboost_0,
                         adaboost_linear, adaboost_lasso, adaboost_elastic_net, adaboost_bayesian_ridge]
    regression_method_names = ["linear_regression", "sigmoid_256", "relu_256", "tanh_256",
                               "xgboost_4", "xgboost_6", "xgboost_10", "xgboost_no_limit",
                              "adaboost_linear", "adaboost_lasso", "adaboost_elastic_net", "adaboost_bayesian_ridge"]
    
    # compute rmses
    
    print "Computing rmses"
    rmses = np.array([mean_rmse(cna_training_data, rna_training_data, proteome_training_data,
                                cna_validation_data, rna_validation_data, proteome_validation_data,
                                 regression_method, num_repeats=1) for regression_method in regression_methods])
        
    # convert to data frame
    rmses = pd.DataFrame(rmses, index=regression_method_names)
    
    print "RMSES:"
    print rmses
    
    # TODO: compare multiple methods and save to file
    fname = "../results/subchallenge_2/{}.csv".format("_".join(regression_method_names))
    print "Saving rmses to {}".format(fname)
    
    # save to file
    rmses.to_csv(fname)

if __name__ == "__main__":
    main()