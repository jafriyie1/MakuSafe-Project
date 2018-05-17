from json_reader import df_reads_specific_json as json_reads
from json_reader import json_concat
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
import numpy as np
import ast


def implementation():
    datafile = json_concat()
    #print(datafile)
    df = pd.concat(datafile)

    #df = df.loc[df['class'].isin([2,13,7])]
    #df = df.sample(50)
    #print(df)
    #datafile = datafile.dropna()
    #print(datafile)

    #df =  pd.DataFrame(datafile)
    #df = pd.read_json(datafile)
    #print(df)

    #df["Features"] = 1
    #print(df.columns)

    #X_features =
    X_df = df["data"]
    y = []
    for count, i in enumerate(X_df):
        #print(i)
        #print(i)
        for x in i:
            #print(x)
            #i[x] = float(x)
            for t in range(len(x)):
                x[t] = float(x[t])
        #print(i)
        #x = np.array(i, dtype=np.float)
        X_df.iloc[count] = i
        #print(x)
        y.append(x)


    #y=np.array([np.array(list(xi), dtype=np.float) for xi in df["data"]])

    #print(y)
    #X_df = np.asarray(y)
    #print(X_df.dtype)
    #X_df.astype(float)

    #X_df = df["data"]
    '''
    for i in range(len(X_df)):
        X_df.iloc[i] = ast.literal_eval(X_df.iloc[i])
    '''

    #print(type(X_df.iloc[1][0][0]))
    X_df = np.array(list(y), dtype=np.float)
    #X_df
    #X_df = np.array(y[0], dtype=np.float)
    print(X_df.shape)
    #nx, ny = X_df.shape
    #X_df = X_df.reshape(1, nx*ny)
    y_df = np.array(list(df["class"]), dtype=np.float)
    print(y_df.shape)

    print("Split....")
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=.2)

    print("SVM model....")
    svc = SVC().fit(X_train, y_train)
    pred = svc.predict(X_test)
    score = str(metrics.accuracy_score(y_test, pred))
    print(score)

    # An example
    import pickle

    print("Extreme Gradient Boosting....")
    xgb = XGBClassifier().fit(X_train, y_train)
    pred = xgb.predict(X_test)
    score = str(metrics.accuracy_score(y_test, pred))
    print(score)

    output = open("save_model.pkl", "wb")
    pickle.dump(xgb, output)

    # To use the model use the following
    pickle_file = open("save_model.pkl", "rb")
    model = pickle.load(pickle_file)

    # Now you can use the model to predict on
    # new data
    model.predict(new_data)




    # 5 to 10 algorithm output

implementation()
