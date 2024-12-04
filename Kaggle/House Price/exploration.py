import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print(df.info())
df = pd.read_csv("train.csv")

first_column = df.pop("SalePrice")
df.insert(0, "SalePrice", first_column)

#print(df["Electrical"].value_counts())
#print(df.info())
df_num = df.select_dtypes(include=["int64", "float64"])
df_obj = df.select_dtypes(include=["object"])

X = df

sns.heatmap(df_num.corr(), vmin=1, vmax=1, annot=True)
plt.show()