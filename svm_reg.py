# Support Vector Regression (SVR) Example in Python
# Author: Senior ML Engineer

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1. Generate synthetic dataset
# -----------------------------
np.random.seed(42)  # For reproducibility
X = np.sort(5 * np.random.rand(100, 1), axis=0)  # Feature values between 0 and 5
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])  # Sine wave + noise

# -----------------------------
# 2. Feature scaling
# -----------------------------
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# -----------------------------
# 3. Train SVR model
# -----------------------------
# RBF kernel is common for nonlinear regression
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

svr_rbf.fit(X_scaled, y_scaled)

# -----------------------------
# 4. Predictions
# -----------------------------
y_pred_scaled = svr_rbf.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# -----------------------------
# 5. Evaluation
# -----------------------------
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# -----------------------------
# 6. Visualization
# -----------------------------
plt.scatter(X, y, color='red', label='Actual Data')
plt.plot(X, y_pred, color='blue', label='SVR Prediction')
plt.title("Support Vector Regression (RBF Kernel)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
