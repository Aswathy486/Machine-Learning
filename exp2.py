import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
cols = ['mpg','cyl','disp','hp','wt','acc','year','org','car']
df = pd.read_csv(url, sep=r"\s+", names=cols)
df = df.replace('?', np.nan).dropna()

# Feature and target
X = df[['disp']]
y = df['mpg']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_lr = lr.predict(X_test)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

pr = LinearRegression()
pr.fit(X_train_poly, y_train)
y_pr = pr.predict(X_test_poly)

# Performance
mse_lr = mean_squared_error(y_test, y_lr)
r2_lr = r2_score(y_test, y_lr)

mse_pr = mean_squared_error(y_test, y_pr)
r2_pr = r2_score(y_test, y_pr)

print("Linear Regression")
print("MSE =", round(mse_lr,2), " R² =", round(r2_lr,3))

print("\nPolynomial Regression")
print("MSE =", round(mse_pr,2), " R² =", round(r2_pr,3))

# Plot
x = np.linspace(X.min(), X.max(), 100).reshape(-1,1)

fig, ax = plt.subplots(1,2,figsize=(12,5),sharey=True)

# Linear Regression Plot
ax[0].scatter(X_test, y_test, color="blue")
ax[0].plot(x, lr.predict(x), color="red")
ax[0].set_title("Linear Regression")
ax[0].text(0.72,0.95,f"MSE={mse_lr:.2f}\nR²={r2_lr:.3f}",
           transform=ax[0].transAxes,bbox=dict(facecolor="white"))

# Polynomial Regression Plot
ax[1].scatter(X_test, y_test, color="blue")
ax[1].plot(x, pr.predict(poly.transform(x)), color="green")
ax[1].set_title("Polynomial Regression")
ax[1].text(0.72,0.95,f"MSE={mse_pr:.2f}\nR²={r2_pr:.3f}",
           transform=ax[1].transAxes,bbox=dict(facecolor="white"))

for a in ax:
    a.set_xlabel("Displacement")
    a.set_ylabel("MPG")
    a.grid(True)

plt.tight_layout()
plt.show()