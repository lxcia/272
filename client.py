from copy import deepcopy
import warnings
from collections import OrderedDict
import utils

import flwr as fl
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import numpy as np
from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

# Class authored by Sophie, Lucia, Bowen
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, args):
        self.training_data, self.testing_data, \
            self.training_labels, self.testing_labels = \
            utils.load_partition(args.partition, args.num_clients, args.data_path, batch_size=args.batch_size)
        self.args = args
        self.net = LogisticRegression(warm_start=True)
        self.net = utils.set_initial_params(self.net)
        print("init")

    def get_parameters(self, config=None): #type: ignore
        return utils.get_model_params(self.net)

    def fit(self, parameters, config): #, config={}):
        self.net = utils.set_parameters(self.net, parameters)
        self.net.fit(self.training_data, self.training_labels)
        return utils.get_model_params(self.net), len(self.training_data), {}

    def evaluate(self, parameters, config={}):
        self.net = utils.set_parameters(self.net, parameters)
        accuracy = self.net.score(self.testing_data, self.testing_labels)
        loss = log_loss(self.testing_labels, self.net.predict_proba(self.testing_data))
        print(f"ACC: {accuracy}")
        return loss, len(self.testing_data), {"accuracy": accuracy}

# Authored by Lucia
def main():
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")

    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        required=False,
        help="Specifies the artificial data partition to be used. \
        Picks partition 0 by default",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=12,
        required=False,
        help="The number of clients to use",
    )

    parser.add_argument(
        "--use_cuda",
        type=bool,
        default=False,
        required=False,
        help="Set to true to use GPU. Default: False",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        required=False,
        help="Batch size to use on each client for training",
    )
    parser.add_argument(
        "-lr",
        type=float,
        default=0.0008,
        required=False,
        help="Learning rate",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=3, #change back to 3
        required=False,
        help="Number of local epochs",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.87,
        required=False,
        help="Momentum for SGD with momentum",
    )
    # Arguments that deal exclusively with data
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the data",
    )
    parser.add_argument(
        "--proximal-mu",
        type=float,
        required=False,
        default=0,
        help="The mu for the proximal term; if this is non-zero, this adds a proximal \
              term to the loss as proposed in the FedProx paper. If this is 0, no proximal \
              term is added."
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        required=False,
        default="val",
        help="Whether to evaluate on the test or the val set. Options are test or val"
    )

    args = parser.parse_args()
    assert args.partition >= 0 and args.partition < args.num_clients, \
        f"Partition {args.partition} is not possible with {args.num_clients} clients."

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
    )

    client = FlowerClient(args)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)


if __name__ == "__main__":
    main()