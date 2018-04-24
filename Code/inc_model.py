import pandas as pd
import json
import ast
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


dataset = pd.read_csv("../Data/incidents_more_cols.csv")
#print(dataset)

data = dataset[["classificationId", "AccelValue"]]
#print(data)

data = data.loc[(data["classificationId"] >=0) & (data["classificationId"] <=6)]
#print(data)
#data = data[:1000]

'''
data_two = data
json_data = data_two["raw"]
print(json_data)
json_data = json_data.tolist()
print(json_data[0])
'''
X_df = data["AccelValue"]
#print(len(X_df))
y_df = data["classificationId"]

for i in range(len(X_df)):
    X_df.iloc[i] = ast.literal_eval(X_df.iloc[i])

'''
for i in range(len(X_df)):
    X_df.iloc[i] = np.array(X_df.iloc[i])
'''

#print(type(x))
#print(x)

#print(X_df)
X_df = np.array(list(X_df), dtype=np.float)
nsamples, nx, ny = X_df.shape
X_df = X_df.reshape(nsamples, nx*ny)
y_df = np.array(list(y_df), dtype=np.float)



X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=.20)

print("Random Forest...")
log_clf = RandomForestClassifier().fit(X_train, y_train)
pred =log_clf.predict(X_test)
score = str(metrics.accuracy_score(y_test, pred))
print(score)
print("On Validation...")
pred = log_clf.predict(X_test)
score = str(metrics.accuracy_score(y_test, pred))
print(score)
print()

#json.loads(json_data[0])
#pd.io.json.json_normalize(data_two.raw.apply(json.loads))
#data_two = data_two["raw"].apply(pd.read_json)
