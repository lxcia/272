import os
import pandas as pd
import torch
import warnings
from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

'''
utils.py
Authored by Bowen, Sophie, Lucia
This file contains functions to load and process training and test 
data from CSVs, get and set model parameters.
'''

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Authored by Sophie
def load_data(data_path):
    '''
    Load and process data from CSVs
    '''
    train_data_path = os.path.join(data_path, "train.csv")
    test_data_path = os.path.join(data_path, "test.csv")

    training_data = pd.read_csv(train_data_path)
    training_data.treatment = training_data.treatment.values - 1
    training_data = training_data.drop(columns=['response_type']).to_numpy()
        
    testing_data = pd.read_csv(test_data_path)
    testing_data.treatment = testing_data.treatment.values - 1
    testing_data = testing_data.drop(columns=['response_type']).to_numpy()

    num_examples = {
        "trainset": len(training_data), "testset": len(testing_data)
    }
    return training_data, testing_data, num_examples


# Authored by Lucia
def load_partition(partition, num_partitions, data_path, batch_size=32):
    '''
    Extracts labels from test and train data
    '''
    training_data, testing_data, num_examples = load_data(data_path)
    n_train = int(num_examples["trainset"])

    training = training_data[:, :-1]
    testing = testing_data[:, :-1]

    training_labels = training_data[:, -1]
    testing_labels = testing_data[:, -1]
    return training, testing, training_labels, testing_labels


# Authored by Bowen, Lucia, Sophie
def get_model_params(model : LogisticRegression) -> LogRegParams:
    '''
    Returns the parameters of a sklearn LogisticRegression model
    '''
    if model.fit_intercept:
        params = [model.coef_, model.intercept_]
    else:
        params = [model.coef_,]
    return params


# Authored by Bowen, Lucia, Sophie
def set_initial_params(model : LogisticRegression):
    '''
    Sets initial parameters as zeros
    '''
    n_classes = 10
    n_features = 13
    model.classes_ = np.array([i for i in range(10)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))
    return model


# Authored by Bowen, Lucia, Sophie
def set_parameters(model : LogisticRegression, parameters: LogRegParams):
    '''
    #TODO
    '''
    model.coef_ = parameters[0]
    if model.fit_intercept:
        model.intercept_ = parameters[1]
    return model
