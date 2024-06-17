# Preprocessing 

import pandas as pd
import numpy as np
import os
import csv

from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

unsw_path = "../data/raw/unsw-nb15"
cic_path = "../data/raw/cic-ids2017"

class Preprocessor:
    def __init__(self):
        self   

    def _unsw_loading(self):
        header = None
        encoding ='utf-8'
        files_list = [os.path.join(f'{unsw_path}/generatedflows', file) for file in os.listdir(f'{unsw_path}/generatedflows') if file.endswith(".csv")]
        data = []
        for file in files_list:
            df_temp = pd.read_csv(file, header=header, low_memory=False, sep=',', encoding=encoding)
            data.append(df_temp)
        unsw = pd.concat(data)
        #UNSW feature names loading
        features = []
        with open (f'{unsw_path}/UNSW-NB15_features.csv') as features_csv:
            filereader = csv.reader(features_csv)
            for row in filereader: 
                features.append(row[1])
        features.pop(0)
        unsw.columns = features
        return unsw
    
    def _cicids_loading(self):
        header = 0
        encoding = 'latin1'
        files_list = [os.path.join(cic_path, file) for file in os.listdir(cic_path) if file.endswith(".csv")]
        data = []
        for file in files_list:
            df_temp = pd.read_csv(file, header=header, low_memory=False, sep=',', encoding=encoding)
            data.append(df_temp)
        cicids = pd.concat(data)
        cicids.columns = [feat.lstrip(' ') for feat in cicids.columns]
        return cicids
    
    def _unsw_cleaning(self):
        ## Drop unnecessary features
        unsw_selected_feats = ["sport", "dsport", "proto", "dur", "Spkts", "Dpkts", "Stime","smeansz",
                                "dmeansz", "Sload", "Dload", "Sintpkt","Dintpkt", "swin", "dwin", 'Label']
        unsw_rmv = [feat for feat in unsw.columns if feat not in unsw_selected_feats]
        unsw = unsw.drop(columns= unsw_rmv)
        return unsw
    
    def _cicids_cleaning(self):
        ## Drop unnecesary features
        cicids_selected_feats = ["Source Port", "Destination Port", "Protocol", "Flow Duration", "Total Length of Fwd Packets", 
                                "Total Length of Bwd Packets", "Timestamp", 'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
                                'Flow Bytes/s', 'Fwd IAT Total', 'Bwd IAT Total','Init_Win_bytes_forward','Init_Win_bytes_backward', 'Label']
        cicids_rmv = [feat for feat in cicids.columns if feat not in cicids_selected_feats]
        # Select columns for the fusion from each df
        cicids = cicids.drop(columns= cicids_rmv)
        # Drop rows where all features are NaN
        cicids.dropna(how='all', axis=0 , inplace=True)
        cicids["Flow Bytes/s"] = cicids["Flow Bytes/s"].fillna(0)
        return cicids

    def _unsw_preprocessing(self,unsw_X):
    ##(sload + dload)/8 -> cicids["Flow bytes/s"]
        unsw_X["Sload"] = unsw_X.Sload + unsw_X.Dload
        unsw_X.rename(columns = {"Sload":"flowbytesps"}, inplace=True)
        unsw_X.flowbytesps.map(lambda x: x/8)
        unsw_X.drop(columns="Dload", inplace = True)
        
    ## Convert epoch time to datetime
        unsw_X["Stime"] = pd.to_datetime(unsw_X["Stime"], unit='s')
        # Extract time component and convert to numeric representation
        unsw_X["Stime"] = unsw_X["Stime"].dt.hour * 3600 + unsw_X["Stime"].dt.minute * 60 + unsw_X["Stime"].dt.second

    ## Categorical features for feature encoding
        categorical_features = ["sport", "dsport","proto"]
        # Convert numerical features to strings
        unsw_X[categorical_features] = unsw_X[categorical_features].astype(str)
    ## Numerical features for feature scaling
        numerical_features = [ feat for feat in unsw_X.columns if feat not in categorical_features]

    ## Scalers and encoders   
        # Feature encoding using Ordinal Encoder
        ord_enc = OrdinalEncoder()
        # Scaling using z-score for numerical features
        zsc = StandardScaler()

    ## Transform
        preprocessor = ColumnTransformer(
            [("standard_scaling", zsc, numerical_features),
            ("feature_encoding", ord_enc, categorical_features)],
            remainder="passthrough"
        )
        unsw_X = preprocessor.fit_transform(unsw_X)
        unsw_X= pd.DataFrame(unsw_X, columns= numerical_features + categorical_features)
        return unsw_X
    
    def _unsw_label(self,unsw_y):
        le = LabelEncoder()
        unsw_y = le.fit_transform(unsw_y)
        unsw_y = pd.DataFrame(unsw_y, columns=["Label"])
        return unsw_y
    
    def _cicids_preprocessing(self,cicids_X):
    ## Time preprocessing
        # Timestamp to Epoch
        timestamps = pd.to_datetime(cicids_X["Timestamp"],format="mixed", dayfirst=True)
        cicids_X["Timestamp"] = (timestamps - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        # Convert epoch time to datetime
        cicids_X["Timestamp"] = pd.to_datetime(cicids_X["Timestamp"], unit='s')
        # Extract time component and convert to numeric representation
        cicids_X["Timestamp"] = cicids_X["Timestamp"].dt.hour * 3600 + cicids_X["Timestamp"].dt.minute * 60 + cicids_X["Timestamp"].dt.second
        
    ## Categorical features for feature encoding
        categorical_features = ["Source Port", "Destination Port","Protocol"]
    ## Numerical features for feature scaling
        numerical_features = [ feat for feat in cicids_X.columns if feat not in categorical_features]
        
    ## Scaling
        # Change infinite values in "Flow Bytes/s" for the maximum value +1 
        max_finite_flowrate = cicids_X["Flow Bytes/s"][np.isfinite(cicids_X["Flow Bytes/s"])].max()
        infinite_flowrate = max_finite_flowrate + 1
        cicids_X["Flow Bytes/s"] = cicids_X["Flow Bytes/s"].replace(np.inf, infinite_flowrate)
        # Z-score for numerical features
        zsc = StandardScaler()
    ## Transform   
        preprocessor = ColumnTransformer(
            [("standard_scaling", zsc, numerical_features)],
            remainder="passthrough"
        )
        cicids_X = preprocessor.fit_transform(cicids_X)
        cicids_X = pd.DataFrame(cicids_X, columns=numerical_features + categorical_features)
        return cicids_X
    
    ## Label encoding
    def _cicids_label(self,cicids_y):
        le = LabelEncoder()
        cicids_y = le.fit_transform(cicids_y)
        cicids_y= pd.DataFrame(cicids_y, columns=["Label"])
        cicids_y.Label= (cicids_y.Label > 0).astype(int)
        return cicids_y
    
    # Reindexing columns to join by column axis
    def _reindex_columns(self, df_X, df_y, new_cols):
        df_X.reset_index(drop=True, inplace=True)
        df_y.reset_index(drop=True, inplace=True)
        df = pd.concat([df_X, df_y], axis=1)
        df = df.reindex(new_cols, axis=1)
        return df

    def run(self):
        # UNSW 
        unsw = self._unsw_loading()
        unsw = self._unsw_cleaning()
        cicids = self._cicids_loading()
        cicids = self._cicids_cleaning()

        # Split in X, y 
        unsw_X = unsw.drop("Label", axis=1)
        unsw_y = unsw["Label"]
        cicids_X = cicids.drop("Label", axis=1)
        cicids_y = cicids["Label"]

        # Split into test and training sets to prevent overfitting
        # UNSW 
        unsw_X_train, unsw_X_test, unsw_y_train, unsw_y_test = train_test_split(unsw_X, unsw_y, test_size=0.2)
        # CIC-IDS2017
        cicids_X_train, cicids_X_test, cicids_y_train, cicids_y_test = train_test_split(cicids_X, cicids_y, test_size=0.2)

        ## Preprocess training data
        unsw_X_train = self._unsw_preprocessing(unsw_X_train)
        unsw_y_train = self._unsw_label(unsw_y_train)
        cicids_X_train = self._cicids_preprocessing(cicids_X_train)
        cicids_y_train = self._cicids_label(cicids_y_train)

        ## Preprocess test data
        unsw_X_test = self._unsw_preprocessing(unsw_X_test)
        unsw_y_test = self._unsw_label(unsw_y_test)
        cicids_X_test = self._cicids_preprocessing(cicids_X_test)
        cicids_y_test = self._cicids_label(cicids_y_test)

        # Column order
        unsw_cols = ["sport", "dsport", "proto", "dur", "Spkts", "Dpkts", "Stime", 
                    "smeansz", "dmeansz", "flowbytesps", "Sintpkt", "Dintpkt", "swin","dwin", "Label"]
        cicids_cols = ["Source Port", "Destination Port", "Protocol", "Flow Duration", "Total Length of Fwd Packets", 
                    "Total Length of Bwd Packets", "Timestamp", "Fwd Packet Length Mean", "Bwd Packet Length Mean",
                    "Flow Bytes/s", "Fwd IAT Total", "Bwd IAT Total", "Init_Win_bytes_forward", "Init_Win_bytes_backward","Label"]
        # Reindexing dataframes
        unsw_train = self._reindex_columns(unsw_X_train, unsw_y_train, unsw_cols)
        unsw_test = self._reindex_columns(unsw_X_test, unsw_y_test, unsw_cols)

        cicids_train = self._reindex_columns(cicids_X_train, cicids_y_train, cicids_cols)
        cicids_test = self._reindex_columns(cicids_X_test, cicids_y_test, cicids_cols)

        # Standarize column names
        fused_data_cols = ["sport","dsport","protocol","duration","spktslength", "dpktslength","timestamp", "spktsmean", "dpktsmean", 
                        "bytesps", "sIAT", "dIAT", "swin", "dwin", "label"]
        unsw_train.columns = fused_data_cols
        unsw_test.columns = fused_data_cols
        cicids_train.columns = fused_data_cols
        cicids_test.columns = fused_data_cols

        # Dataframe fusion
        fused_data_train= pd.concat([unsw_train, cicids_train])
        fused_data_test = pd.concat([unsw_test, cicids_test])
        #fused_data_train.to_csv("data/training/data_train.csv")
        #fused_data_test.to_csv("data/test/data_test.csv")

        # Shuffle data
        fused_shuffled_data_train= shuffle(fused_data_train, random_state=42)
        fused_shuffled_data_test= shuffle(fused_data_test, random_state=42)

        # Split the dataset into two halves for each federated device
        split_point = len(fused_shuffled_data_train) // 2
        train_data_part1 = fused_shuffled_data_train.iloc[:split_point]
        train_data_part2 = fused_shuffled_data_train.iloc[split_point:]
        split_point = len(fused_shuffled_data_test) // 2
        test_data_part1 = fused_shuffled_data_test.iloc[:split_point]
        test_data_part2 = fused_shuffled_data_test.iloc[split_point:]

        # Save the splitted datasets to CSV files for distribution
        train_data_part1.to_csv('../data/site1/train_data.csv', index=False)
        train_data_part2.to_csv('../data/site2/train_data.csv', index=False)
        test_data_part1.to_csv('../data/site1/test_data.csv', index=False)
        test_data_part2.to_csv('../data/site2/test_data.csv', index=False)

def main():
    preprocessor = Preprocessor()
    preprocessor.run()

if __name__ == "__main__":
    main()