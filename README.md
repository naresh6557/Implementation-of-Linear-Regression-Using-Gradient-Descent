# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset (Population vs Profit) and initialize parameters (θ, α, iterations).
2.Define the cost function to measure error between predicted and actual values.

3.Update θ values iteratively using Gradient Descent until the cost is minimized.

4.Use the optimized θ to predict profit and plot the regression line with the data. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: NARESH KUMAR R
RegisterNumber: 212224040213
*/
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
import warnings

# 🔇 Suppress convergence warnings
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# Sample dataset: [Square Feet, Number of Rooms, Age of House]
X = np.array([
    [1000, 3, 10],
    [1500, 4, 5],
    [800, 2, 20],
    [1200, 3, 8],
    [2000, 5, 3],
    [950, 2, 15],
    [1750, 4, 4],
    [1600, 4, 6]
])

# Target 1: House prices in $1000s
y_price = np.array([150, 200, 120, 170, 300, 130, 270, 250])

# Target 2: Number of occupants
y_occupants = np.array([3, 4, 2, 3, 5, 2, 5, 4])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_price_train, y_price_test = train_test_split(X_scaled, y_price, test_size=0.2, random_state=42)
_, _, y_occ_train, y_occ_test = train_test_split(X_scaled, y_occupants, test_size=0.2, random_state=42)

# 🔁 SGD Regressors with increased max_iter and lower tol
price_model = SGDRegressor(max_iter=5000, tol=1e-4, random_state=42)
occupant_model = SGDRegressor(max_iter=5000, tol=1e-4, random_state=42)

# Fit the models
price_model.fit(X_train, y_price_train)
occupant_model.fit(X_train, y_occ_train)

# Predict on new input
sample_house = np.array([[1400, 3, 7]])
sample_scaled = scaler.transform(sample_house)

predicted_price = price_model.predict(sample_scaled)[0]
predicted_occupants = occupant_model.predict(sample_scaled)[0]

print(f"Predicted House Price: ${predicted_price * 1000:.2f}")
print(f"Predicted Number of Occupants: {round(predicted_occupants)}")

# Optional: Plot actual vs predicted price
y_price_pred = price_model.predict(X_test)
plt.scatter(y_price_test, y_price_pred, color='blue')
plt.plot([min(y_price_test), max(y_price_test)],
         [min(y_price_test), max(y_price_test)], 'r--', label='Perfect Prediction')
plt.xlabel("Actual Price ($1000s)")
plt.ylabel("Predicted Price ($1000s)")
plt.title("Actual vs Predicted House Price")
plt.grid(True)
plt.legend()
plt.show()
```

## Output:
<img width="930" height="643" alt="image" src="https://github.com/user-attachments/assets/04ebcaad-df06-4451-8f3a-fb73f5e29fb3" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
