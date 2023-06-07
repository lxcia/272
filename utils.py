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


def load_data(partition, num_partitions, data_path):
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


def load_partition(partition, num_partitions, data_path, batch_size=32):
    training_data, testing_data, num_examples = load_data(partition, num_partitions, data_path)
    n_train = int(num_examples["trainset"])

    training = training_data[:, :-1]
    testing = testing_data[:, :-1]

    training_labels = training_data[:, -1]
    testing_labels = testing_data[:, -1]
    return training, testing, training_labels, testing_labels


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
