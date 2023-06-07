# Authored by Sophie
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from joblib import dump

df_test = pd.read_csv("final_data/clinic 0/test.csv")
df_train = pd.read_csv("final_data/clinic 0/train.csv")
for i in range(1,15):
    # Aggregate train
    file_name_train = "final_data/"+str("clinic ")+str(i)+"/train.csv"
    curr_df_train = pd.read_csv(file_name_train)
    df_train = pd.concat([df_train,curr_df_train])

    # Aggregate test
    file_name_test = "final_data/"+str("clinic ")+str(i)+"/test.csv"
    curr_df_test = pd.read_csv(file_name_test)
    df_test = pd.concat([df_test,curr_df_test])
# Train shape
print(df_train.shape)

# Test shape
print(df_test.shape)

y = df_train['treatment'].values
print(df_train.columns)
X = df_train.drop(['treatment','Unnamed: 0','response_type'], axis=1).values
#39.795*comm rounds
#39795
clf = LogisticRegression(random_state=0, max_iter = 39795, multi_class = 'multinomial').fit(X,y)
dump(clf, "logreg_baseline.joblib")

with open("independent_log_reg.txt",'w') as f:
    for i in range(15,25):
        file_name_test = "final_data/"+str("clinic_")+str(i)+".csv"
        df_test = pd.read_csv(file_name_test)
        y_test = df_test['treatment'].values
        X_test = df_test.drop(['treatment','response_type'], axis=1).values
        preds = clf.predict_proba(X_test)
        np.save("clinic_"+str(i)+"_preds.npy",preds)
        
        #preds_joined = ",".join(preds)
        # f.write("clinic_"+str(i)+",")
        # f.write(str(preds))
        # f.write("\n")
