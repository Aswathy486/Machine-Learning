import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing

# 1. Load Data
housing = fetch_california_housing(as_frame=True)
data = housing.frame

# Using 'AveRooms' as chosen in your variable
X = data[["AveRooms"]].values
y = data["MedHouseVal"].values

# --- CRITICAL FIX: FEATURE SCALING ---
# Manual standardization so Gradient Descent does not explode
X_mean = np.mean(X)
X_std = np.std(X)
X_scaled = (X - X_mean) / X_std
# -------------------------------------

np.random.seed(42)

# 2. Train/Test Split (using scaled features)
shuffled_indices = np.random.permutation(len(X_scaled))
test_set_size = int(len(X_scaled) * 0.2)

test_indices = shuffled_indices[:test_set_size]
train_indices = shuffled_indices[test_set_size:]

X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Add Bias Column
X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]


# 3. Algorithms
def gradient_descent(X, y, alpha=0.01, iterations=5000):
    m = len(y)
    theta = np.zeros(X.shape[1])  # Initialize weights to 0

    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1 / m) * X.T.dot(errors)
        theta = theta - alpha * gradient

    return theta


def normal_equation(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


# Train models
theta_gd = gradient_descent(X_train_bias, y_train, alpha=0.01, iterations=5000)
theta_ne = normal_equation(X_train_bias, y_train)

# Predict on Test Data (to match your metrics evaluation)
y_pred_gd = X_test_bias.dot(theta_gd)
y_pred_ne = X_test_bias.dot(theta_ne)


# 4. Evaluation
def evaluate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return mse, r2


mse_gd, r2_gd = evaluate_metrics(y_test, y_pred_gd)
mse_ne, r2_ne = evaluate_metrics(y_test, y_pred_ne)

print("\nGRADIENT DESCENT RESULTS (Scaled Feature)")
print(f"Intercept: {theta_gd[0]:.4f}, Slope: {theta_gd[1]:.4f}")
print(f"MSE: {mse_gd:.4f}, R² Score: {r2_gd:.4f}\n")

print("NORMAL EQUATION RESULTS (Scaled Feature)")
print(f"Intercept: {theta_ne[0]:.4f}, Slope: {theta_ne[1]:.4f}")
print(f"MSE: {mse_ne:.4f}, R² Score: {r2_ne:.4f}\n")

# ----------------------------------------------------
# 5. Visualization (Plotting Test Data to match Test Metrics)
# ----------------------------------------------------
plt.figure(figsize=(14, 5))
sort_idx = np.argsort(X_test.flatten())

# Subplot 1: Gradient Descent Fit
plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, color="blue", alpha=0.2, label="Test Data")
plt.plot(
    X_test[sort_idx],
    X_test_bias[sort_idx].dot(theta_gd),
    color="red",
    linewidth=2,
    label="GD Line",
)
plt.title(f"Gradient Descent Fit\nMSE: {mse_gd:.4f} | R² Score: {r2_gd:.4f}")
plt.xlabel("AveRooms (Standardized)")
plt.ylabel("MedHouseVal")
plt.legend()

# Subplot 2: Normal Equation Fit
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color="blue", alpha=0.2, label="Test Data")
plt.plot(
    X_test[sort_idx],
    X_test_bias[sort_idx].dot(theta_ne),
    color="green",
    linewidth=2,
    label="Normal Eq Line",
)
plt.title(f"Normal Equation Fit\nMSE: {mse_ne:.4f} | R² Score: {r2_ne:.4f}")
plt.xlabel("AveRooms (Standardized)")
plt.ylabel("MedHouseVal")
plt.legend()

plt.tight_layout()
plt.show()