# Kaggle challenge: https://www.kaggle.com/competitions/playground-series-s4e10
# Kaggle score: 0.94357

# Gradient boost
import catboost as cb
from catboost import CatBoostRegressor, Pool, EShapCalcType, EFeaturesSelectionAlgorithm, CatBoostClassifier

# Data Frame
import pandas as pd

# Feature engineering 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# read csv
df = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#encode
le = LabelEncoder()
columns_to_encode = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]

for col in columns_to_encode:
	le.fit(df[col])
	df[col] = le.transform(df[col])
	test_data[col] = le.transform(test_data[col])

# split features - target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# split data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# train model
model = CatBoostRegressor(iterations=100, learning_rate=0.1, random_seed=42, depth=6, verbose=0)
model.fit(X_train, y_train)

# predict data output
test_predict = model.predict(test_data)
test_predict_df = pd.DataFrame({"id": test_data.id, "loan_status": test_predict})
test_predict_df.to_csv("output.csv", index=False)
