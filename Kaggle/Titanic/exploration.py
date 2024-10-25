import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
'''
# Read Data
gs = pd.read_csv("gender_submission.csv")
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

# Encode function
encoder = LabelEncoder()

# Cleaning
	# Label Encoding
train["Sex"] = encoder.fit_transform(train["Sex"])
train["Embarked"] = encoder.fit_transform(train["Embarked"])

trainNum = train.select_dtypes(include=['float64', 'int32', "int64"])

trainNum.plot(subplots=True)
plt.show()
'''
df = pd.read_csv("submission.csv")
print(df)