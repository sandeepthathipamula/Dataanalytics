import pandas as pd

data = pd.read_csv("dataset.csv")
data.dropna(inplace=True)
z_scores = (data - data.mean()) / data.std()
data = data[(z_scores < 3).all(axis=1)]
data.drop_duplicates(inplace=True)
data.to_csv("prepared_dataset.csv", index=False)