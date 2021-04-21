import argparse
import pickle
import logging
import os
import pandas as pd

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

import numpy as np
np.set_printoptions(suppress=True)


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

ROOT_PATH = os.path.join("//net","dataset")
ROOT_PATH = os.path.join("//var","cache","fscache","audiffre")
PATH_TO_CLEANED_DATA = os.path.join(ROOT_PATH, "pp","data","cleaned_data")
PATH_TO_CLEANED_DATA = os.path.join(ROOT_PATH, "cleaned_data")

def get_all_indexes(self,  year = "2018", embedding = "300LG"):
    ID_train = np.load(os.path.join(PATH_TO_CLEANED_DATA, "indexes_train.npy"))
    ID_valid = np.load(os.path.join(PATH_TO_CLEANED_DATA, "indexes_valid.npy"))
    ID_test = np.load(os.path.join(PATH_TO_CLEANED_DATA, "indexes_test.npy"))

    all_ids = np.concatenate([ID_train, ID_valid, ID_test])

    return all_ids


class DataLoader() :
    def __init__(self, year = "2018", embedding = "300LG",with_image = True):

        if year == "2018":
            if embedding =="300LG":
                if with_image :
                    path_X = os.path.join(PATH_TO_CLEANED_DATA,"X_full_2018.npy")
                else :
                    path_X = os.path.join(PATH_TO_CLEANED_DATA,"X_cleaned_20180813.npy")
                path_Y = os.path.join(PATH_TO_CLEANED_DATA,"Y_cleaned_20180813.npy")
        elif year == "2019":
            if embedding == "300LG":
                if with_image :
                    path_X = os.path.join(PATH_TO_CLEANED_DATA,"X_full_2019.npy")
                else :
                    path_X = os.path.join(PATH_TO_CLEANED_DATA,"X_cleaned_20190916.npy")
                path_Y = os.path.join(PATH_TO_CLEANED_DATA,"Y_cleaned_20190916.npy")
            


        if with_image:
            X =pd.read_feather(path_X)
            indexes = X.C_index
            X.index = indexes
            self.X = X.drop(columns = ["C_index"])

        else :
            self.X = pd.read_pickle(path_X)
        self.Y = pd.read_pickle(path_Y)


    def get_train(self):
        ID = np.load(os.path.join(PATH_TO_CLEANED_DATA, "indexes_train.npy"))
        X = self.X.loc[ID].values
        Y = self.Y.loc[ID].values

        is_valid = (np.sum(np.isnan(X),axis=1)==0)
        X=X[is_valid]
        Y=Y[is_valid]


        return X, Y

    def get_valid(self):
        ID = np.load(os.path.join(PATH_TO_CLEANED_DATA, "indexes_valid.npy"))
        X = self.X.loc[ID].values
        Y = self.Y.loc[ID].values

        is_valid = (np.sum(np.isnan(X),axis=1)==0)
        X=X[is_valid]
        Y=Y[is_valid]

        return X, Y

    def get_test(self):
        ID = np.load(os.path.join(PATH_TO_CLEANED_DATA, "indexes_test.npy"))
        X = self.X.loc[ID].values
        Y = self.Y.loc[ID].values


        is_valid = (np.sum(np.isnan(X),axis=1)==0)
        X=X[is_valid]
        Y=Y[is_valid]

        return X, Y

    def get_all(self):
        X = self.X
        Y = self.Y


        is_valid = (np.sum(np.isnan(X),axis=1)==0)
        X=X[is_valid]
        Y=Y[is_valid]

        return X.values, Y.values, X.index



    def get_test_index(self):
        ID = np.load(os.path.join(PATH_TO_CLEANED_DATA, "indexes_test.npy"))
        X = self.X.loc[ID]
        is_valid = (np.sum(np.isnan(X),axis=1)==0)
        X=X[is_valid]
        return X.index


    def get_topclass(self):
        ID = np.load(os.path.join("/net","dataset","pp","data",'TopClass_20180813_indexes_test.npy'))
        X = self.X.loc[ID].values
        Y = self.Y.loc[ID].values

        is_valid = (np.sum(np.isnan(X),axis=1)==0)
        X=X[is_valid]
        Y=Y[is_valid]

        return X, Y

    def get_q(self, Q):
        X = self.X.loc[[Q]].values
        Y = self.Y.loc[[Q]].values

        is_valid = (np.sum(np.isnan(X),axis=1)==0)
        X=X[is_valid]
        Y=Y[is_valid]

        return X, Y


    def get_prediction_transformation(self):
        prop_transform = np.load(os.path.join(PATH_TO_CLEANED_DATA,"properties_2019_to_2018.npy"))
        return prop_transform


    def get_properties(self):
        properties = np.load(os.path.join(PATH_TO_CLEANED_DATA,"properties_2018.npy"), allow_pickle=True)
        return properties


    def get_nlabels(self):
        return self.Y.shape[1]

    def get_ndims(self):
        return self.X.shape[1]

    def clean(self):
        self.X = None
        self.Y = None



if __name__ == "__main__":

    #data = DataLoader(year = "2019")
    #X_train, Y_train = data.get_train()

    s = os.path.relpath((os.path.join("..","data","classes.pickle")),start=os.path.curdir)
    print(s)

    Z= pickle.load( open(s,"rb" ) )
