import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

class LocalModel(nn.Module):
    def __init__(self, input_size=13, layer_sizes=[10,20,5], output_size=10):
        super(LocalModel, self).__init__()
        layer_sizes = [input_size] + layer_sizes + [output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if(i != len(layer_sizes) - 1):
                layers.append(nn.ReLU())
        self.all_layers = nn.Sequential(*layers)

    def forward(self, x):
       layer_output = self.all_layers(x)
       return(nn.functional.softmax(layer_output, dim=1))

    # def process_input(self, train_data):
    #     # Load the training data from CSV
    #     train_labels = train_data['treatment'].values
    #     train_data = train_data.drop(['treatment','Unnamed: 0','response_type'], axis=1).values
    #
    #     # Convert data to PyTorch tensors
    #     train_data = torch.from_numpy(train_data).float()
    #     train_labels = torch.from_numpy(train_labels).long()
    #     return train_data, train_labels

    def train(self, train_data, train_labels, epochs=10000, learn=0.000001):
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learn)

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            outputs = self.forward(train_data.float()) #fix train data dtype
            loss = loss_func(outputs, train_labels.long())

            # Backward and optimize
            loss.backward()
            optimizer.step()

            if (epoch) % 100 == 0:
                max_probs, predictions = torch.max(outputs, 1)
                accuracy = sum([1 for i in range(len(predictions)) if predictions[i] == train_labels[i]])/len(outputs)
                print("accuracy: "+str(accuracy))




    def test(self, test_data, test_labels):
        loss_func = nn.CrossEntropyLoss()
        # test_data, test_labels = self.process_input(test_csv)
        outputs = self.forward(test_data.float())
        max_probs, predictions = torch.max(outputs, 1)
        accuracy = sum([1 for i in range(len(predictions)) if predictions[i] == test_labels[i]])/len(outputs)
        loss = loss_func(outputs, test_labels.long())
        print("accuracy: "+str(accuracy))
        return loss, accuracy


# def main():
#     model = LocalModel()
#     df = pd.read_csv("~/Downloads/clinic_datasets_super_easy_mode-3/clinic_0.csv")
#     df_sample = df.sample(n=int(len(df)/10))
#
#     for i in range(1,15):
#         file_name = "~/Downloads/clinic_datasets_super_easy_mode-3/clinic_"+str(i)+".csv"
#         curr_df = pd.read_csv(file_name)
#         curr_df_sample = curr_df.sample(n=int(len(curr_df)/2))
#         df_sample = pd.concat([df_sample, curr_df_sample])
#
#     print("Training data shape")
#     print(df_sample.shape)
#
#     #train_data, train_labels = model.process_input(df_sample)
#
#     #model.train(train_data, train_labels)
#     #outputs = model.test("~/Downloads/clinic_datasets/clinic_1.csv")
#
#     y = df_sample['treatment'].values
#     X = df_sample.drop(['treatment','Unnamed: 0','response_type'], axis=1).values
#     print(df_sample.columns)
#     clf = LogisticRegression(random_state=0, max_iter = 100000, multi_class = 'multinomial').fit(X,y)
#     print(clf.score(X,y))
#
# if __name__ == '__main__':
#     main()