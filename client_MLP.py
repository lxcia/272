import utils
import flwr as fl
import torch
import argparse
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

'''
client_MLP.py
Authored by Sophie, Bowen, Lucia
This file contains the implementation of 2-layer Multi-Layer Perceptron model run
at each client. To make this model the architecture used in the implementation, change
the name to "client.py" An alternative architecture can currently be found in "client.py"
'''

# Class authored by Sophie, Debugging help from Bowen
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, args):
        self.training_data, self.testing_data, \
            self.training_labels, self.testing_labels = \
            utils.load_partition(args.partition, args.num_clients, args.data_path, batch_size=args.batch_size)
        
        self.label_binarizer = LabelBinarizer().fit(np.arange(10))
        self.training_labels = self.label_binarizer.transform(self.training_labels)
        self.testing_labels = self.label_binarizer.transform(self.testing_labels)

        self.args = args
        self.net = Sequential()
        self.net.add(Dense(64, input_shape=(13,), activation='relu'))
        self.net.add(Dense(64, activation='relu'))
        self.net.add(Dense(10, activation='softmax'))
        self.net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("init")


    def get_parameters(self, config):
        return self.net.get_weights()


    def fit(self, parameters, config):
        self.net.set_weights(parameters)
        self.net.fit(self.training_data, self.training_labels, epochs=25, batch_size=32, steps_per_epoch=8)
        return self.net.get_weights(), len(self.training_data), {}


    def evaluate(self, parameters, config):
        self.net.set_weights(parameters)
        loss, accuracy = self.net.evaluate(self.testing_data, self.testing_labels)
        return loss, len(self.testing_data), {"accuracy": float(accuracy)}


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
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)


if __name__ == "__main__":
    main()