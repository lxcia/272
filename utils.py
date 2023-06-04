import os
import pandas as pd
import torch
import random
import numpy
from torch.utils.data import DataLoader
import warnings
from torch.utils.data import Dataset
from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Create a custom Dataset type for DataLoader
# class MyData(Dataset):
#     def __init__(self, df):
#         self.df = df
#
#
#     def __len__(self):
#         return self.df.shape[0]
#
#     def __getitem__(self, index):
#         info = torch.from_numpy(self.df[:, :-1])  # Hormonal levels
#         label = torch.from_numpy(self.df[:, -1])  # BC
#         return info, label


def load_data(partition, num_partitions, data_path):
    """Load training and test set."""
    #assert len(os.listdir(data_path)) == num_partitions, \
        #f"Data path {data_path} contains {len(os.listdir(data_path))} datasets but there are {num_partitions} partitions."
    print("load data")
    train_data_path = os.path.join(data_path, "train.csv")
    test_data_path = os.path.join(data_path, "test.csv")
    val_data_path = os.path.join(data_path, "val.csv")

    # training_data = numpy.loadtxt(train_data_path, delimiter=',', skiprows=1)
    # testing_data = numpy.loadtxt(test_data_path, delimiter=',', skiprows=1)
    # val_data = numpy.loadtxt(val_data_path, delimiter=',', skiprows=1)
    training_data = pd.read_csv(train_data_path)
    training_data.treatment = training_data.treatment.values - 1
    training_data = training_data.drop(columns=['response_type']).to_numpy()
        
    testing_data = pd.read_csv(test_data_path)
    testing_data.treatment = testing_data.treatment.values - 1
    testing_data = testing_data.drop(columns=['response_type']).to_numpy()
        
    val_data = pd.read_csv(val_data_path)
    val_data.treatment = val_data.treatment.values - 1
    val_data = val_data.drop(columns=['response_type']).to_numpy()
    print(f'Training data shape: {training_data.shape}')

    num_examples = {
        "trainset": len(training_data), "testset": len(testing_data), "valset": len(val_data)
    }
    print("done loading data")
    #for checking dimensions
    # print(testing_data.shape, val_data.shape, testing_data.shape)
    # assert(0==1)
    return training_data, testing_data, val_data, num_examples


# DONT NEED TO CALL BC ALREADY ST UP CSVS
def load_partition(partition, num_partitions, data_path, batch_size=32):
    training_data, testing_data, val_data, num_examples = load_data(partition, num_partitions, data_path)

    n_train = int(num_examples["trainset"])
    print("n_train:", n_train)

    training = training_data[:, :-1]
    testing = testing_data[:, :-1]
    val = val_data[:, :-1]

    training_labels = training_data[:, -1]
    val_labels = val_data[:, -1]
    testing_labels = testing_data[:, -1]
    print(f'Training data shape after tensorification: {training.shape}')
    print("done load_partition")

    return training, val, testing, training_labels, val_labels, testing_labels
    # MAYBE FIX ABOVE LABEL GENEARATION LAST RETURN
#
# def get_model_params(model):
#     """Returns a model's parameters."""
#     return [val.cpu().numpy() for _, val in model.state_dict().items()]

def get_model_params(model : LogisticRegression) -> LogRegParams:
    """Returns the paramters of a sklearn LogisticRegression model"""
    if model.fit_intercept:
        params = [model.coef_, model.intercept_]
    else:
        params = [model.coef_,]
    return params

def set_initial_params(model : LogisticRegression):
    """
    Sets initial parameters as zeros
    """
    n_classes = 10
    n_features = 13
    model.classes_ = np.array([i for i in range(10)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))
    return model

def set_parameters(model : LogisticRegression, parameters: LogRegParams):
    model.coef_ = parameters[0]
    if model.fit_intercept:
        model.intercept_ = parameters[1]
    return model
# def difference_models_norm_2(model_1, model_2):
#     """
#     Return the norm 2 difference between the two model parameters
#     Copied from https://epione.gitlabpages.inria.fr/flhd/federated_learning/FedAvg_FedProx_MNIST_iid_and_noniid.html
#     """
#
#     tensor_1 = list(model_1.parameters())
#     tensor_2 = list(model_2.parameters())
#
#     norm = sum([torch.sum((tensor_1[i] - tensor_2[i]) ** 2)
#                 for i in range(len(tensor_1))])
#
#     return norm