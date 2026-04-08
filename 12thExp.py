# DA 12 Experiment:
# Apply Predictive analytics for Weather forecasting.

# Code


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load your dataset (assuming a standard weather.csv)
# If you don't have one, you can create a dummy dataframe for testing
data = {
    "Humidity": [60, 65, 70, 75, 80, 85, 90, 95],
    "Wind_Speed": [10, 12, 15, 14, 18, 20, 22, 25],
    "Pressure": [1012, 1011, 1010, 1009, 1008, 1007, 1006, 1005],
    "Temperature": [30, 28, 27, 25, 24, 22, 21, 19],
}
df = pd.DataFrame(data)

# 2. Features (X) and Target (y)
X = df[["Humidity", "Wind_Speed", "Pressure"]]
y = df["Temperature"]

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Initialize and Train the Model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# 5. Make Predictions
predictions = model.predict(X_test)

# 6. Evaluate Performance
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Sample Prediction
new_data = np.array([[72, 13, 1010]])  # Humidity=72, Wind=13, Pressure=1010
predicted_temp = model.predict(new_data)
print(f"Predicted Temperature: {predicted_temp[0]:.2f}°C")
