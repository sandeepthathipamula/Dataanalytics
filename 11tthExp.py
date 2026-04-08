# DA 11 Experiment:
# Perform Predictive analytics on Product Sales data
# Code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Create a Sample Dataset
data = {
    "Day": np.arange(1, 31),
    "Sales": [
        200,
        210,
        190,
        220,
        230,
        250,
        245,
        260,
        280,
        300,
        310,
        305,
        325,
        340,
        350,
        370,
        360,
        380,
        395,
        410,
        420,
        430,
        425,
        450,
        470,
        480,
        490,
        510,
        520,
        530,
    ],
}
df = pd.DataFrame(data)

# 2. Prepare Features (X) and Target (y)
X = df[["Day"]]  # Features must be a 2D array
y = df["Sales"]

# 3. Split Data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Initialize and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make Predictions
predictions = model.predict(X_test)

# 6. Predict Sales for a Future Day (e.g., Day 35)
future_day = np.array([[35]])
future_prediction = model.predict(future_day)
print(f"Predicted Sales for Day 35: {future_prediction[0]:.2f}")

# 7. Visualize the Results
plt.scatter(X, y, color="blue", label="Actual Sales")
plt.plot(X, model.predict(X), color="red", label="Regression Line (Trend)")
plt.title("Product Sales Prediction")
plt.xlabel("Day")
plt.ylabel("Sales Volume")
plt.legend()
plt.show()
