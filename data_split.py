import os
import pandas as pd

for i in range(0,25):
    folder_name = "clinic "+str(i)
    folder_path = os.path.join("final_data",folder_name)
    train_path = os.path.join(folder_path,"train.csv")
    val_path = os.path.join(folder_path,"val.csv")
    test_path = os.path.join(folder_path,"test.csv")
    os.mkdir(folder_path)
    all_data = pd.read_csv("final_data/"+"clinic_"+str(i)+".csv")
    print("Total samples: "+str(len(all_data)))
    test = all_data.sample(n=int(all_data.shape[0]*0.2), random_state=42)
    print("Test samples: "+str(len(test)))
    train = all_data.drop(test.index)
    print("Train samples: "+str(len(train)))
    test.to_csv(test_path)
    train.to_csv(train_path)