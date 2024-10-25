import catboost as cb
from catboost import CatBoostRegressor, Pool, EShapCalcType, EFeaturesSelectionAlgorithm, CatBoostClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

#read
df = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#split
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

#encode
le = LabelEncoder()
columns_to_encode = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]

for col in columns_to_encode:
	le.fit(X[col])
	X[col] = le.transform(X[col])
	test_data[col] = le.transform(test_data[col])

#split data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
model = CatBoostClassifier(iterations=100, learning_rate=0.1, random_seed=42, depth=6, verbose=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred_series = pd.Series(y_pred, index=X_test.index, name="Prediction")
print(metrics.accuracy_score(y_test, y_pred))



test_predict = model.predict(test_data)
test_predict_df = pd.DataFrame({"id": test_data.id, "loan_status": test_predict})
test_predict_df.to_csv("output.csv", index=False)