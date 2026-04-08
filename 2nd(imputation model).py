import pandas as pd
from sklearn.impute import SimpleImputer

data = pd.read_csv("dataset.csv")

imputer = SimpleImputer(strategy="mean")
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

data_imputed.to_csv("imputed_data.csv", index=False)