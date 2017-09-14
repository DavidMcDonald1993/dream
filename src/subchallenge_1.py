import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from fancyimpute import SimpleFill, KNN, SoftImpute, IterativeSVD, MICE, MatrixFactorization, NuclearNormMinimization
from keras.models import Model
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.objectives import binary_crossentropy, mean_squared_error
from keras.callbacks import EarlyStopping
import keras.backend as K
import tensorflow as tf

# root mean squared error with three masks
def rmse (original_data, y_pred, y_true): 
    # rsme prediction and ground truth
    rmse_no_mask = np.sqrt(mse(y_true, y_pred))
    
    # ignore all zeros in the ground truth data
    no_zeros = y_true > 0
    rmse_no_zeros = np.sqrt(mse(y_true[no_zeros], y_pred[no_zeros]))
    
    # ignore zeros and only consider data that was originally nan in the training data
    nan_no_zeros = np.isnan(original_data) & (y_true > 0)
    rmse_nan_no_zeros = np.sqrt(mse(y_true[nan_no_zeros], y_pred[nan_no_zeros]))
    
    # concatenate all three results
    return np.array([rmse_no_mask, rmse_no_zeros, rmse_nan_no_zeros])

# compute mean rmse across a number of repreats
def mean_rmse(data, imputation_method, y_true, num_repeats=1, **kwargs):
    
    imputed_predictions = [imputation_method(data, **kwargs) for i in range(num_repeats)]
    
    rmses = np.array([rmse(data, imputed_prediction, y_true) for imputed_prediction in imputed_predictions])

    return rmses.mean(axis=0)

# imputation methods
# impute with sample mean
def sample_mean(data, **kwargs):
    fill = SimpleFill(fill_method="mean")
    return fill.complete(data)

# impute with knn-3
def knn_3(data, **kwargs):
    fill = KNN(k=3, verbose=0)
    return fill.complete(data)

# impute with knn-5
def knn_5(data, **kwargs):
    fill = KNN(k=5, verbose=0)
    return fill.complete(data)

# knn for any k
def knn(data, k, **kwargs):
    fill = KNN(k=k, verbose=0)
    return fill.complete(data)

# softimpute from fancyimpute package
def soft_impute(data, **kwargs):
    fill = SoftImpute(verbose=0)
    return fill.complete(data)

# removing to focus on optimising soft impute

# # iterativeSVD from fancy impute package
# def iterative_SVD(data, **kwargs):
#     fill = IterativeSVD(verbose=0)
#     return fill.complete(data)

# # MICE for fancyimpute package
# def mice(data, **kwargs):
#     fill = MICE(verbose=0)
#     return fill.complete(data)

# modified autoencoder that does not propagate error from missing values
def modified_autoencoder(data, num_hidden=[32], dropout=0.1, **kwargs):
    
    # dimensionality of data
    num_proteins, num_features = data.shape
    
    # to normalise the data we must impute 
    mean_imputer = SimpleFill(fill_method="mean")
    data_imputed = mean_imputer.complete(data)
    
    # standard scaling for normalisation
    standard_scaler = StandardScaler()
    data_imputed_and_scaled = standard_scaler.fit_transform(data_imputed)
    
    # replace all missing values with 0 so they do not contribute to input
    data_imputed_and_scaled[np.isnan(data)] = 0
    
    # maintain nan in target data so we know which outputs should not prodice any error
    data_scaled_with_nan = np.array([[data_imputed_and_scaled[i, j] if ~np.isnan(data[i, j]) else np.nan
                                     for j in range(num_features)] for i in range(num_proteins)])
    
    # custom MSE that only produces error on non-nan terms
    def custom_MSE(y_true, y_pred):
    
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        # mask for targets that are not nan
        mask = ~tf.is_nan(y_true)

        # apply the mask to targets and output of network and then compute MSE with what remains
        y_true = tf.boolean_mask(tensor=y_true, mask=mask)
        y_pred = tf.boolean_mask(tensor=y_pred, mask=mask)

        return mean_squared_error(y_true, y_pred)

    
    # construct model
    x = Input(shape=(num_features,))
    
    # first fully connected layer layer
    y = Dense(num_hidden[0], activation="relu")(x)
    y = BatchNormalization()(y)
    y = Dropout(dropout)(y)

    # all remaining fully connected layers
    for h in num_hidden[1:] + num_hidden[-2::-1]:
        y = Dense(h, activation="relu")(y)
        y = BatchNormalization()(y)
        y = Dropout(dropout)(y)
    
    # output -- no activation function 
    y = Dense(num_features, activation="linear")(y)
    autoencoder = Model(x, y)
    autoencoder.compile(optimizer="adam", loss=custom_MSE)
    early_stopping = EarlyStopping(monitor="loss", patience=1000, min_delta=0)
    # train model
    autoencoder.fit(data_imputed_and_scaled, data_scaled_with_nan, 
                    verbose=0, epochs=10000, batch_size=100, callbacks=[early_stopping])
    # predict data
    prediction = autoencoder.predict(data_imputed_and_scaled)
    
    # reverse normalise and return
    return standard_scaler.inverse_transform(prediction)

# PCA and then autoencoder
def pca_autoencoder(data, num_hidden=[32], dropout=0.1, pca_dim=64, **kwargs):
    
    
    # dimensionality of data
    num_proteins, num_features = data.shape
    
    #construct model
    x = Input(shape=(pca_dim,))
    y = Dropout(1e-8)(x)
    for h in num_hidden + num_hidden[-2::-1]:
        y = Dense(h, activation="relu")(y)
        y = BatchNormalization()(y)
        y = Dropout(dropout)(y)
    y = Dense(pca_dim)(y)
    
    autoencoder = Model(x, y)
    autoencoder.compile(optimizer="adam", loss="mse")
    
    
    # project with pca
    mean_imputer = SimpleFill()
    data_imputed = mean_imputer.complete(data)
    pca = PCA(n_components=pca_dim)
    data_transformed = pca.fit_transform(data_imputed)
    early_stopping = EarlyStopping(monitor="loss", patience=100, min_delta=0)
    autoencoder.fit(data_transformed, data_transformed, 
                    verbose=0, epochs=10000, batch_size=100, callbacks=[early_stopping])
    
    prediction = autoencoder.predict(data_transformed)
    
    return pca.inverse_transform(prediction)



def main():
    
    print "Loading data"

    # training data
    dfs = [pd.read_csv("../data/sub_challenge_1/data_obs_{}.txt".format(i), 
                    header=0, index_col=0, sep="\t") for i in range(1, 11)]

    # ground truth
    ground_truth_table = pd.read_csv("../data/sub_challenge_1/data_true.txt", 
                    header=0, index_col=0, sep="\t")

    # conver from data frame ot numpy array
    datas = [df.values for df in dfs]
    ground_truth = ground_truth_table.values

    # list of imputation tecniques
    imputation_methods = [sample_mean, knn_3, knn_5, soft_impute, 
                          modified_autoencoder, pca_autoencoder]
#     imputation_methods = [sample_mean]
    
    print "Computing rmse"
    
    # iterate over all training data and imputation methods and compute mean rmse for num repeats
    rmses = np.array([[mean_rmse(data, imputation_method, ground_truth, num_repeats=1) for data in datas] 
                      for imputation_method in imputation_methods])
    
    print "Saving rmse to file"
    
    # save to file
    np.savetxt(X=rmses[:,:,0], 
               fname="../results/subchallenge_1/{}_rmses_no_mask.csv".format(imputation_methods), delimiter=",")
    np.savetxt(X=rmses[:,:,1], 
               fname="../results/subchallenge_1/{}_rmses_ignore_zeros.csv".format(imputation_methods), delimiter=",")
    np.savetxt(X=rmses[:,:,2], 
               fname="../results/subchallenge_1/{}_rmses_only_nan_ignore_zeros.csv".format(imputation_methods), 
               delimiter=",")
    

if __name__ == "__main__":
    main()