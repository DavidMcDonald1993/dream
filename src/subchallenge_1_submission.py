import os
import numpy as np
import pandas as pd
from fancyimpute import KNN

from sklearn.preprocessing import StandardScaler

def load_data(filename):
    
    df = pd.read_csv(filename, sep="\t", header=0, index_col=0, )
    
    return df.values, df.index, df.columns

def impute(data):
    
    knn = KNN(k=5, verbose=0)
    
    return knn.complete(data)

def main(directory):

    directory = directory.replace("*", "")

    for filename in os.listdir(directory):
        
        data, index, columns = load_data(os.path.join(directory, filename))
        
        imputed_prediction = impute(data)
        confidences = np.ones(imputed_prediction.shape)
        
        columns = ["patientId{}".format(i+1) for i in range(data.shape[1])]
        index = ["protein{}".format(i+1) for i in range(data.shape[0])]
        
        imputed_prediction = pd.DataFrame(imputed_prediction, index = index, columns=columns)
        imputed_prediction.index.name = "proteinID"
        confidences = pd.DataFrame(confidences, index = index, columns=columns)
        confidences.index.name = "proteinID"
        
        file_id =  [int(s) for s in filename if s.isdigit()][0]
        output_prediction_filename = "../output/predictions_{}.tsv".format(file_id)
        output_confidence_filename = "../output/confidence_{}.tsv".format(file_id)
        
        imputed_prediction.to_csv(output_prediction_filename, sep="\t")
        confidences.to_csv(output_confidence_filename, sep="\t")

if __name__ == "__main__":
    directory = "../evaluation_data/*"
    main(directory) 