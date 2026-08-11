import ssl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score

ssl._create_default_https_context = ssl._create_unverified_context

data = fetch_20newsgroups(subset="all",
        categories=["comp.graphics", "sci.space"],
        remove=("headers", "footers", "quotes"))

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

# Multinomial NB
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)
m_acc = accuracy_score(y_test,
        MultinomialNB().fit(X_train_cv, y_train).predict(X_test_cv))

# Bernoulli NB
cv = CountVectorizer(binary=True)
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)
b_acc = accuracy_score(y_test,
        BernoulliNB().fit(X_train_cv, y_train).predict(X_test_cv))

print("Multinomial NB:", round(m_acc*100,2), "%")
print("Bernoulli NB :", round(b_acc*100,2), "%")

acc = [m_acc*100, b_acc*100]
plt.bar(["Multinomial", "Bernoulli"], acc)

for i, v in enumerate(acc):
    plt.text(i, v+1, f"{v:.2f}%", ha="center")

plt.ylim(0,100)
plt.title("Naive Bayes Comparison")
plt.ylabel("Accuracy (%)")
plt.savefig("exp7/naive_bayes_comparison.png")
plt.show()