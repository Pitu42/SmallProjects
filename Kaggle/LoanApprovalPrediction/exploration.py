import catboost as cb
from catboost import CatBoostRegressor, Pool, EShapCalcType, EFeaturesSelectionAlgorithm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

#read
df = pd.read_csv("train.csv")

#split
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

#encode
le = LabelEncoder()
columns_to_encode = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]

for col in columns_to_encode:
	X[col] = le.fit_transform(X[col])

# split data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
#model = CatBoostRegressor(iterations=100, learning_rate=0.1, random_seed=42, depth=6, verbose=0)
#model.fit(X_train, y_train)

#y_pred = model.predict(X_test)

#y_pred_series = pd.Series(y_pred, index=X_test.index, name="Prediction")
#result = pd.concat([y_test, y_pred_series], axis=1)
#print(result.to_markdown())

feature_names = ['F{}'.format(i) for i in range(X_train.shape[1])]
train_pool = Pool(X_train, y_train, feature_names=feature_names)
test_pool = Pool(X_test, y_test, feature_names=feature_names)
model = CatBoostRegressor(iterations=1000, random_seed=0)
summary = model.select_features(
    train_pool,
    eval_set=test_pool,
    features_for_select='0-99',
    num_features_to_select=10,
    steps=3,
    algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
    shap_calc_type=EShapCalcType.Regular,
    train_final_model=True,
    logging_level='Silent',
    plot=True
)