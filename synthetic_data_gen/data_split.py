# Authored by Sophie

import os
import pandas as pd

# Generate 80-20 train-validation splits for each clinic
for i in range(0,25):
    # Set up paths for the train and validation data
    folder_name = "clinic "+str(i)
    folder_path = os.path.join("final_data",folder_name)
    train_path = os.path.join(folder_path,"train.csv")
    test_path = os.path.join(folder_path,"test.csv")
    os.mkdir(folder_path)
    # Read in current clinic
    all_data = pd.read_csv("final_data/"+"clinic_"+str(i)+".csv")
    # Split data and print number of samples in train, test, and total
    print("Total samples: "+str(len(all_data)))
    test = all_data.sample(n=int(all_data.shape[0]*0.2), random_state=42)
    print("Test samples: "+str(len(test)))
    train = all_data.drop(test.index)
    print("Train samples: "+str(len(train)))
    # Write the splits to new csv files
    test.to_csv(test_path)
    train.to_csv(train_path)