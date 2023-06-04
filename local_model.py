import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

class LocalModel(nn.Module):
    def __init__(self, input_size=13, layer_sizes=[32, 64, 32], output_size=10):
        super(LocalModel, self).__init__()
        # Set up array of all layer sizes
        layer_sizes = [input_size] + layer_sizes + [output_size]
        layers = []
        # Architecture has a ReLU between every linear layer
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if(i != len(layer_sizes) - 1):
                layers.append(nn.ReLU())
        # Use sequential to combine all the layers
        self.all_layers = nn.Sequential(*layers)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        layer_output = self.all_layers(x)
        # Apply softmax to output of layers
        return (nn.functional.softmax(layer_output, dim=1))

    #def forward(self, x):
        # Pass the input tensor through each of our operations
        #for layer in self.all_layers:
            #if isinstance(layer, nn.Linear):
                # input = input.to(torch.float32)
                #x = x.to(torch.float32)
                #x = layer(x)
            #else:
                #x = x.to(torch.float32)
                #x = layer(x)
        #return nn.functional.softmax(x, dim=1)
    #
    # def process_input(self, train_data):
    #     print("processing input")
    #     # Load the training data from CSV
    #     train_labels = np.array(train_data['treatment'].values)
    #     train_data = np.array(train_data.drop(['treatment','response_type'], axis=1).values)
    #
    #     # Convert the numpy arrays to tensors
    #     train_data = torch.from_numpy(train_data, requires_grad=True).float()
    #     train_labels = torch.from_numpy(train_labels).long()
    #     return train_data, train_labels

    def train(self, train_data, train_labels, epochs=10000, learn=0.000001):
        print("training")
        # Set up loss and optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learn)

        # Train for the desired number of epochs
        for epoch in range(epochs):
            # optimizer.zero_grad()
            # Forward pass
            train_data = train_data.to(torch.float32)
            train_data.requires_grad_() #please for the love of god work
            outputs = self.forward(train_data)
            #train_labels = torch.argmax(train_labels, dim=0)
            train_labels = train_labels.long()
            outputs = outputs.double()
            outputs = torch.tensor(outputs)
            #print(" OUTPUTS ")
            #print(outputs)
            #print("TRAIN LABELS")
            #print(train_labels)
            train_labels = torch.tensor(train_labels)
            loss = self.loss_func(outputs, train_labels)
            loss.requires_grad_()

            # Backward pass
            loss.backward()
            optimizer.step()

            # Print accuracy every 100 epochs
            if (epoch) % 100 == 0:
                max_probs, predictions = torch.max(outputs, 1)
                accuracy = sum([1 for i in range(len(predictions)) if predictions[i] == train_labels[i]])/len(outputs)
                print("accuracy: "+str(accuracy))

    # def train(self, train_data, train_labels, batch_size=32, epochs=10000, learn=0.000001):
    #     print("training")
    #     # Set up loss and optimizer
    #     loss_func = nn.CrossEntropyLoss()
    #     optimizer = torch.optim.SGD(self.parameters(), lr=learn)
    #
    #     # Determine the total number of samples
    #     num_samples = len(train_data)
    #
    #     # Train for the desired number of epochs
    #     for epoch in range(epochs):
    #         optimizer.zero_grad()
    #
    #         # Iterate over batches
    #         for i in range(0, num_samples, batch_size):
    #             # Get the current batch
    #             batch_data = train_data[i:i + batch_size]
    #             batch_labels = train_labels[i:i + batch_size]
    #
    #             # Forward pass
    #             batch_data = batch_data.to(torch.float32)
    #             outputs = self.forward(batch_data)
    #
    #             batch_labels = torch.argmax(batch_labels, dim=0)  # Convert one-hot to class indices
    #             batch_labels = batch_labels.long()  # Convert to torch.long
    #             outputs = outputs.double()
    #
    #             loss = loss_func(outputs, batch_labels)
    #
    #             # Backward pass
    #             loss.backward()
    #             optimizer.step()
    #
    #         # Print accuracy every 100 epochs
    #         if (epoch + 1) % 100 == 0:
    #             _, predictions = torch.max(outputs, 1)
    #             accuracy = (predictions == batch_labels).sum().item() / batch_labels.size(0)
    #             print("Accuracy: {:.2f}%".format(accuracy * 100))


    def test(self, test_data, test_labels):
        print("testing")

        # Predict on test set
        test_data = test_data.to(torch.float32)
        test_data.requires_grad_()  # please for the love of god work
        outputs = self.forward(test_data)
        max_probs, predictions = torch.max(outputs, 1)
        ground_truth_labels = test_labels
        ground_truth_labels = ground_truth_labels.long()
        outputs = outputs.double()
        outputs = torch.tensor(outputs)

        # Print accuracy on train set
        accuracy = np.average(np.equal(np.array(predictions), np.array(test_labels)))
        loss = self.loss_func(outputs, ground_truth_labels)
        print("accuracy: "+str(accuracy))
        print(f'loss: {loss}')
        return loss, accuracy


def main():
    model = LocalModel(layer_sizes=[1])

    # For testing the local model create an aggregate of the clinics that combines 10% of patients from each clinic
    df = pd.read_csv("~/Downloads/clinic_datasets_sophie_hormone_generation/clinic_0.csv")
    df.treatment = df.treatment.values - 1
    print(df.columns)
    df_sample = df.sample(n=int(len(df)/10))

    for i in range(1,15):
        file_name = "~/Downloads/clinic_datasets_sophie_hormone_generation/clinic_"+str(i)+".csv"
        curr_df = pd.read_csv(file_name)
        curr_df.treatment = curr_df.treatment.values - 1
        curr_df_sample = curr_df.sample(n=int(len(curr_df)/2))
        df_sample = pd.concat([df_sample, curr_df_sample])

    print("Training data shape")
    print(df_sample.shape)

    # Train the models
    train_data, train_labels = model.process_input(df_sample)
    model.train(train_data, train_labels)

    # Run model on test set
    #outputs = model.test("~/Downloads/clinic_datasets/clinic_1.csv")

    # Compare performance with that of a logistic regression
    y = df_sample['treatment'].values
    X = df_sample.drop(['treatment','response_type'], axis=1).values
    print(df_sample.columns)
    clf = LogisticRegression(random_state=0, max_iter = 100000, multi_class = 'multinomial').fit(X,y)
    print(clf.score(X,y))

if __name__ == '__main__':
    main()



