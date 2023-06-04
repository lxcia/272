from copy import deepcopy
import warnings
from collections import OrderedDict
import utils

import flwr as fl
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from local_model import LocalModel


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, args):
        self.training_data, self.testing_data, self.val_data, self.labels = utils.load_partition(
            args.partition, args.num_clients, args.data_path, batch_size=args.batch_size
        )
        self.args = args
        self.net = LocalModel()
        print("init")

    # Config is never used in the below functions but is expected by the
    # numpy client
    def get_parameters(self, config={}):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config={}):
        print("fit")
        self.set_parameters(parameters)
        #def train(self, train_data, train_labels, epochs=10000, learn=0.000001):
        self.net.train(
            self.training_data,
            self.labels,
            # TODO: CHECK THAT EPOCCHS AND LR ALIGN WITH DEFAULT
            epochs=self.args.local_epochs,
            learn=self.args.lr
        )  # Adjust number of local updates b/t communication rounds
        return self.get_parameters(config={}), len(self.training_data.dataset), {}

    def evaluate(self, parameters, config={}):
        # assert self.args.eval_dataset == 'test' or self.args.eval_dataset == 'val'
        dataset = self.testing_data # if self.args.eval_dataset == 'test' else self.val_data
        self.set_parameters(parameters)
        # (self, test_csv):
        loss, accuracy = self.net.test(self.net, dataset)
        print(loss, accuracy)
        return loss, len(dataset), {"accuracy": accuracy}


#def run_baseline(args):
    #training_data, testing_data = utils.load_partition(
        #0, 1, args.data_path, batch_size=args.batch_size
    #)

    #net = Net()
    #train(
        #net,
        #training_data,
        #epochs=args.local_epochs,
        #testloader=testing_data, # Evaluates performance every local epoch
    #)
    #loss, accuracy = test(net, testing_data)
    #print(f"Loss: {loss}, Accuracy: {accuracy}")


def main():
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--baseline",
        action='store_true',
        help="Whether to run a baseline experiment where a single client \
        trains on the entire dataset (i.e., a non-federated approach)"
    )
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
        "--toy",
        type=bool,
        default=False,
        required=False,
        help="Set to true to quicky run the client using only 10 datasamples. \
        Useful for testing purposes. Default: False",
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
        default=3,
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

    #if args.baseline:
        #run_baseline(args)
    #else:
    # REMOVED BASELINE CASE (commented above)

    client = FlowerClient(args)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)


if __name__ == "__main__":
    main()