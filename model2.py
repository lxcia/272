import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

class MLP(nn.Module):
    def __init__(self, input_size=7, hidden_size1=16, hidden_size2=32, hidden_size3=64, output_size=10):
        super(MLP, self).__init__()
        self.all_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),
            nn.Linear(hidden_size3, output_size)
        )
        

    def forward(self, x):
       layer_output = self.all_layers(x)
       return(nn.functional.softmax(layer_output, dim=1))

    def process_input(self, train_data):
        # Load the training data from CSV
        #train_data = pd.read_csv(train_csv)
        train_labels = train_data['ocp'].values.astype(np.long)
        train_data = train_data.drop('Unnamed: 0', axis=1)
        train_data = train_data.drop('ocp', axis=1).values
       
        # Convert data to PyTorch tensors
        train_data = torch.from_numpy(train_data).float()
        train_labels = torch.from_numpy(train_labels)
        return train_data, train_labels

    def train(self, train_data, train_labels, num_epochs=100000, learning_rate=0.0000001):
        # train_data, train_labels = self.process_input(train_csv)
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Forward pass
            outputs = self.forward(train_data)
            loss = loss_func(outputs, train_labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                max_probs, predictions = torch.max(outputs, 1)
                print(train_labels[0].type)
                print(predictions[0].type)
                num_matched = [1 for i in range(len(predictions)) if predictions[i] == train_labels[i]]
                accuracy = sum(num_matched)/len(outputs)
                print("accuracy: "+str(accuracy))




    def test(self, test_csv):
        test_data, test_labels = self.process_input(test_csv)
        outputs = self.forward(test_data)
        max_probs, predictions = torch.max(outputs, 1)
        num_matched = [1 for i in range(len(predictions)) if predictions[i] == test_labels[i]]
        accuracy = sum(num_matched)/len(outputs)
        print("accuracy: "+str(accuracy))
        return outputs


def main():
    model = MLP()
    df = pd.read_csv("hormones_only.csv")
    train_data, train_labels = model.process_input(df)

    #model.train(train_data, train_labels)
    #outputs = model.test("~/Downloads/clinic_datasets/clinic_1.csv")

    #X,y = model.process_input("~/Downloads/clinic_datasets/clinic_5.csv")
    y = df['ocp'].values
    X = df.drop('ocp',axis=1).values
    clf = LogisticRegression(random_state=0, max_iter = 100000, multi_class = 'multinomial').fit(X,y)
    print(clf.score(X,y))

if __name__ == '__main__':
    main()



