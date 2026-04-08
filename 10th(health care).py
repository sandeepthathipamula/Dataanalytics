import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("healthcare_data.csv")

print(data.head())

print(data.describe())

print(data.nunique())

print(data["category"].value_counts())

plt.hist(data["age"], bins=10)
plt.show()