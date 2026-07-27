import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load and split data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
models = {
    "MLE": LogisticRegression(penalty=None, max_iter=10000),
    "MAP (L1)": LogisticRegression(penalty="l1", solver="liblinear"),
    "MAP (L2)": LogisticRegression(penalty="l2")
}

acc, coef = [], []
for name, model in models.items():
    model.fit(X_train, y_train)
    acc.append(accuracy_score(y_test, model.predict(X_test)) * 100)
    coef.append(model.coef_[0])

# Plot graphs
fig, ax = plt.subplots(2, 2, figsize=(10, 7))

ax[0,0].bar(models.keys(), acc, color=["red","blue","green"])
ax[0,0].set_title("Accuracy Comparison")
ax[0,0].set_ylabel("Accuracy (%)")
ax[0,0].set_ylim(80,100)

titles = ["MLE Weights", "MAP L1 Weights", "MAP L2 Weights"]
colors = ["red", "blue", "green"]

for i in range(3):
    r, c = (0,1) if i==0 else (1,i-1)
    ax[r,c].plot(coef[i], marker='o', color=colors[i])
    ax[r,c].axhline(0, color="black", linestyle="--")
    ax[r,c].set_title(titles[i])
    ax[r,c].set_xlabel("Feature Index")
    ax[r,c].set_ylabel("Weight")

plt.tight_layout()
plt.savefig("exp4.png")
plt.show()