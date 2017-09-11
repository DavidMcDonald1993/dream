# regression model
from keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

from fancyimpute import KNN

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error as mse

from sklearn.linear_model import LinearRegression

def preprocess_data():
    
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

    # use complete data as training data
    cna_training_data = cna_data_frame.loc[complete_proteins, patients].values
    rna_training_data = rna_data_frame.loc[complete_proteins, patients].values
    proteome_training_data = proteome_data_frame.loc[complete_proteins, patients].values

    # standard scaling
    cna_scaler = StandardScaler()
    rna_scaler = StandardScaler()
    proteome_scaler = StandardScaler()

    cna_training_data = cna_scaler.fit_transform(cna_training_data)
    rna_training_data = rna_scaler.fit_transform(rna_training_data)
    proteome_training_data = proteome_scaler.fit_transform(proteome_training_data)
    
    
    # return training data scalers for inverse scaling
    return cna_training_data, rna_training_data, proteome_training_data, cna_scaler, rna_scaler, proteome_scaler

def rmse (y_pred, y_true): 
    # rsme prediction and ground truth
    return np.sqrt(mse(y_true, y_pred))

# compute mean rmse across a number of repreats
def mean_rmse(cna_training_data, rna_training_data, proteome_training_data, regression_method, num_repeats=1, **kwargs):
    
    abundance_predictions = [regression_method(cna_training_data, rna_training_data, proteome_training_data, **kwargs) 
                             for i in range(num_repeats)]
    
    rmses = np.array([rmse(abundance_prediction, proteome_training_data) 
                      for abundance_prediction in abundance_predictions])

    return rmses.mean(axis=0)

def linear_regression(cna_training_data, rna_training_data, proteome_training_data):
    
    print "Building linear model"
    
    linear_model = LinearRegression()
    
    print "Appending training matrices for linear model"
    
    # append data for linear regression
    cna_rna_training_data_append = np.append(cna_training_data, rna_training_data, axis=1)
    
    print "Fitting linear model"
    
    linear_model.fit(cna_rna_training_data_append, proteome_training_data)
    
    abundance_predictions = linear_model.predict(cna_rna_training_data_append)
    
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
                               num_hidden=[64], dropout=0.1, activation="relu", reg=1e-3):
    
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
    abundance_predictions = regression_model.predict([cna_training_data, rna_training_data])
    
    return abundance_predictions

def main():
    
    print "Preprocessing data"

    # get training data
    cna_training_data, rna_training_data, proteome_training_data, cna_scaler, rna_scaler, proteome_scaler = preprocess_data()
    
    #list of regression methods
    regression_methods = [linear_regression, deep_non_linear_regression]
    
    # compute rmses
    
    print "Computing rmses"
    rmses = np.array([mean_rmse(cna_training_data, rna_training_data, proteome_training_data,
                                 regression_method, num_repeats=1) for regression_method in regression_methods])
    
    print "RMSES:"
    print rmses
    
    # TODO: compare multiple methods and save to file
    fname = "../results/subchallenge_2/{}.csv".format(regression_methods) 
    print "Saving rmses to {}".format(fname)
    
#     # save to file
    np.savetxt(X=rmses, fname=fname, delimiter=",")

if __name__ == "__main__":
    main()