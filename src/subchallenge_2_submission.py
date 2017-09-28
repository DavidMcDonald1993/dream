import numpy as np
import pandas as pd

import pickle

from keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.models import model_from_json

from fancyimpute import KNN

def preprocess_data(cna_data_frame, rna_data_frame, data_dim=50):
    
    # find common proteins and patients across all datasets
    proteins = proteome_data_frame.index.intersection(cna_data_frame.index).intersection(rna_data_frame.index)
    patients = proteome_data_frame.columns.intersection(cna_data_frame.columns).intersection(rna_data_frame.columns)

    # locate common data and return as numpy array
    cna_data = cna_data_frame.loc[proteins, patients].values
    rna_data = rna_data_frame.loc[proteins, patients].values
    
    print "PCA to reduced to fixed dimension"
    
    # PCA to reduce unknown dimension to fixed dimension
    cna_pca = PCA(n_components=data_dim)
    rna_pca = PCA(n_components=data_dim)
    
    cna_data = cna_pca.fit_transform(cna_data)
    rna_data = rna_pca.fit_transform(rna_data)
    
    print "scaling data"

    # standard scaling
    cna_scaler = StandardScaler()
    rna_scaler = StandardScaler()

    cna_data = cna_scaler.fit_transform(cna_data)
    rna_data = rna_scaler.fit_transform(rna_data)
    
    # return training data scalers for inverse scaling
    return cna_data, rna_data

def load_model(model_json_file, model_weights_file):
    
    json_file = open(model_json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_weights_file)
    
    return loaded_model
    
def main():
    
    cna_data_frame = pd.read_csv("/evaluation_data/prospective_ova_CNA_median_sort_common_gene_11859.txt", 
                header=0, index_col=0, sep="\t")
    rna_data_frame = pd.read_csv("/evaluation_data/prospective_ova_rna_seq_sort_common_gene_15121.txt", 
                    header=0, index_col=0, sep="\t")
    
    cna_data, rna_data, cna_pca, rna_pca, cna_scaler, rna_scaler = preprocess_data()
    
    # load scalers for proteomics data
    with open('pca_scaler.pkl', 'rb') as f:
        proteome_pca, proteome_scalar = pickle.load(f)
    
    model = load_model("model.json", "model_weights.h5")
    
    abundance_predictions = model.predict([cna_data, rna_data])
    
    # unscale
    abundance_predictions = proteome_scalar.inverse_transform(abundance_predictions)
    # un-PCA
    abundance_predictions = proteome_pca.inverse_transform(abundance_predictions)
    
    confidences = np.ones(abundance_predictions.shape)
    
    output_prediction_filename = "/output/predictions.tsv"
    output_confidence_filename = "/output/confidence.tsv"
    
    columns = ["patientId{}".format(i+1) for i in range(len(cna_data_frame.columns))]
    index = ["protein{}".format(i+1) for i in range(len(cna_data_frame.index))]
    
    abunance_predictions = pd.DataFrame(abunance_predictions, columns=columns, index=index)
    abundance_predictions.index.name = "proteinID"
    confidences = pd.DataFrame(confidences, columns=columns, index=index)
    confidences.index.name = "proteinID"
    
    abundance_predictions.to_csv(output_prediction_filename, sep="\t")
    confidences.to_csv(output_confidence_filename, sep="\t")
    
if __name__ == "__main__":
    main()