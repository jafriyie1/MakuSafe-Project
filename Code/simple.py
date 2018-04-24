import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier


df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


size = len(df.columns) -1
size_two = len(test_df.columns) -1

X_df = df.iloc[:, 0:size]
y_df = df.iloc[:, size]

print(y_df.value_counts())

"""
51 0's
15 1's
2 2's

Can change into a binomial problem if needed (incident and non incident)


"""

X_test_df = test_df.iloc[:, 0:size_two]
y_test_df = test_df.iloc[:, size_two]



X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=.20)

print("Logistic Regression...")
log_clf = LogisticRegression().fit(X_train, y_train)
pred =log_clf.predict(X_test)
score = str(metrics.accuracy_score(y_test, pred))
print(score)
print("On Validation...")
pred = log_clf.predict(X_test_df)
score = str(metrics.accuracy_score(y_test_df, pred))
print(score)
print()

print("SVM model....")
svc = SVC().fit(X_train, y_train)
pred = svc.predict(X_test)
score = str(metrics.accuracy_score(y_test, pred))
print(score)
print("On Validation....")
pred = svc.predict(X_test_df)
score = str(metrics.accuracy_score(y_test_df, pred))
print(score)
print()


print("Random Forest....")
rf = RandomForestClassifier().fit(X_train, y_train)
pred = rf.predict(X_test)
score = str(metrics.accuracy_score(y_test, pred))
print(score)
print("On Validation....")
pred = rf.predict(X_test_df)
score = str(metrics.accuracy_score(y_test_df, pred))
print(score)
print()

print("Extreme Gradient Boosting....")
xgb = XGBClassifier().fit(X_train, y_train)
pred = xgb.predict(X_test)
score = str(metrics.accuracy_score(y_test, pred))
print(score)
"""
print("On Validation....")
pred = xgb.predict(X_test_df)
score = str(metric.accuracy_score(y_test_df, pred))
print(score)
"""
