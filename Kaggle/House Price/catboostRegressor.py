import pandas as pd
from sklearn import metrics

df = pd.read_csv("train.csv")

print(df.iloc[:,[17]])

# Encode obj
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_obj = df.select_dtypes(include=["object"])

for column in df_obj.columns:
	df[column] = le.fit_transform(df[column])

# Test variables
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]



from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

from catboost import CatBoostRegressor, Pool, EShapCalcType, EFeaturesSelectionAlgorithm

model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, verbose=0)

feature_names = ['F{}'.format(i) for i in range(train_X.shape[1])]
train_pool = Pool(train_X, train_y, feature_names=feature_names)
test_pool = Pool(test_X, test_y, feature_names=feature_names)

model = CatBoostRegressor(iterations=1000, random_seed=0)
summary = model.select_features(
    train_pool,
    eval_set=test_pool,
    features_for_select='0-70',
    num_features_to_select=1,
    steps=3,
    algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
    shap_calc_type=EShapCalcType.Regular,
    train_final_model=True,
    logging_level='Silent',
    plot=True
)
print(summary)
print(df[17].info())
#model.fit(X_train, y_train)

#y_pred = model.predict(X_test)
#print(metrics.r2_score(y_test, y_pred))


