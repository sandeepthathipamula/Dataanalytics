import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

data = pd.read_csv("time_series_data.csv", index_col="Date", parse_dates=True)

model = ARIMA(data, order=(5, 1, 0))
fit_model = model.fit()

forecast = fit_model.predict(start=len(data), end=len(data) + 10)

print(forecast)