import ssl, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

ssl._create_default_https_context = ssl._create_unverified_context

data = fetch_20newsgroups(subset='train',
    categories=['comp.graphics', 'sci.space'],
    remove=('headers', 'footers', 'quotes'))

vec = CountVectorizer(max_features=1000, stop_words='english')
counts = np.asarray(vec.fit_transform(data.data).sum(axis=0)).ravel()
N, V = counts.sum(), len(counts)

df = pd.DataFrame({"Word": vec.get_feature_names_out(), "MLE": counts / N, "Count": counts})
for a in [0.1, 2, 5]:
    df[f"MAP {a}"] = (counts + a - 1) / (N + V * (a - 1))

top = df.nlargest(5, "Count")
print(top[["Word", "MLE", "MAP 0.1", "MAP 2", "MAP 5"]])

x = np.arange(5); w = 0.2
plt.figure(figsize=(8,4))
plt.bar(x, top["MLE"], w, label="MLE")
for i, a in enumerate([0.1, 2, 5], 1):
    plt.bar(x + i*w, top[f"MAP {a}"], w, label=f"MAP {a}")

plt.xticks(x + 1.5*w, top["Word"])
plt.title("MLE vs MAP")
plt.ylabel("Probability")
plt.legend()
plt.tight_layout()
plt.show()