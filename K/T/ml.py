import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Clean training Data
encoder = LabelEncoder()
train["Sex"] = encoder.fit_transform(train["Sex"])
train["Embarked"] = encoder.fit_transform(train["Embarked"])
train["Cabin"] = encoder.fit_transform(train["Cabin"])
print(train)

X = train.drop(["Survived","Name", "Ticket"] , axis=1)
y = train["Survived"]

#Clean testing Data
print(test)
test["Sex"] = encoder.fit_transform(test["Sex"])
test["Embarked"] = encoder.fit_transform(test["Embarked"])
test["Cabin"] = encoder.fit_transform(test["Cabin"])
test = test.drop(["Name", "Ticket"] , axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators = 100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn import metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

test_predict = clf.predict(test)
print(test)
print(test_predict)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': test_predict})
output.to_csv('submission.csv', index=False)

'''
y_pred_series = pd.Series(y_pred, index=X_test.index, name='Predicted')
result = pd.concat([X_test, y_test, y_pred_series], axis=1)
result = result.drop("PassengerId", axis=1)
print(result)
print(train.info()
	)
'''