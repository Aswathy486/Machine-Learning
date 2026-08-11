import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
data = pd.read_csv("diabetes.csv")
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Function to train and evaluate
def evaluate(Xtr, Xte):
    model = LogisticRegression(max_iter=1000)
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    return [
        accuracy_score(y_test, pred),
        precision_score(y_test, pred),
        recall_score(y_test, pred),
        f1_score(y_test, pred)
    ]

# Without Feature Scaling
score1 = evaluate(X_train, X_test)

# With Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
score2 = evaluate(X_train, X_test)

# Print Results
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

print("Without Feature Scaling")
for m, s in zip(metrics, score1):
    print(f"{m}: {s:.3f}")

print("\nWith Feature Scaling")
for m, s in zip(metrics, score2):
    print(f"{m}: {s:.3f}")

# Bar Chart
x = range(len(metrics))
plt.figure(figsize=(7,4))

bars1 = plt.bar([i-0.2 for i in x], score1, width=0.4,
                label="Without Scaling")
bars2 = plt.bar([i+0.2 for i in x], score2, width=0.4,
                label="With Scaling")

# Add values on bars
for bar in bars1:
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02,
             f"{bar.get_height():.2f}",
             ha="center")

for bar in bars2:
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02,
             f"{bar.get_height():.2f}",
             ha="center")

plt.xticks(x, metrics)
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.title("Performance Comparison of Logistic Regression")
plt.legend()
plt.savefig("exp6.png")
plt.show()