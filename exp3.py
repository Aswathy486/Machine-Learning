import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# Load Diabetes dataset
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Ridge Regression
ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

# Lasso Regression
lasso = LassoCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5, random_state=42)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)

# Calculate Metrics
models = ["Linear", "Ridge", "Lasso"]

mse = [
    mean_squared_error(y_test, lr_pred),
    mean_squared_error(y_test, ridge_pred),
    mean_squared_error(y_test, lasso_pred)
]

r2 = [
    r2_score(y_test, lr_pred),
    r2_score(y_test, ridge_pred),
    r2_score(y_test, lasso_pred)
]

# Print Results
print("Linear Regression")
print("MSE:", mse[0])
print("R2 Score:", r2[0])

print("\nRidge Regression")
print("Best Alpha:", ridge.alpha_)
print("MSE:", mse[1])
print("R2 Score:", r2[1])

print("\nLasso Regression")
print("Best Alpha:", lasso.alpha_)
print("MSE:", mse[2])
print("R2 Score:", r2[2])

# ---------------- Visualization ----------------

colors = ['#6C5CE7', '#FD9644', '#20BF6B']

plt.figure(figsize=(12,5))

# MSE Graph
plt.subplot(1,2,1)
bars = plt.bar(models, mse, color=colors, edgecolor='black', width=0.5)
plt.title("Mean Squared Error (MSE)\n[Lower is Better]", fontsize=15, fontweight='bold')
plt.ylabel("MSE Value")
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    plt.text(bar.get_x()+bar.get_width()/2,
             bar.get_height()+30,
             f"{bar.get_height():,.2f}",
             ha='center', fontsize=11, fontweight='bold')

# R2 Graph
plt.subplot(1,2,2)
bars = plt.bar(models, r2, color=colors, edgecolor='black', width=0.5)
plt.title(r"$R^2$ Score (Variance Explained)"+"\n[Higher is Better]",
          fontsize=15, fontweight='bold')
plt.ylabel(r"$R^2$ Score")
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    plt.text(bar.get_x()+bar.get_width()/2,
             bar.get_height()+0.005,
             f"{bar.get_height():.4f}",
             ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()