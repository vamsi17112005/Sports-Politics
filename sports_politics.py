import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



data = pd.read_csv("bbc_data.csv")
data.columns = data.columns.str.lower()

# keeping only sport and politics
data = data[data["labels"].isin(["sport", "politics"])]

print("Class distribution:")
print(data["labels"].value_counts())


X = data["data"]
y = data["labels"]


# small cleanup so punctuation doesn't create extra features
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


X = X.apply(clean_text)


# stratify ensures both classes appear in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain size:", len(X_train))
print("Test size:", len(X_test))
print("\nTest distribution:")
print(y_test.value_counts())


results = {}


def run_experiment(name, vectorizer, model):

    # fit only on training data (important to avoid leakage)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)

    print("\n==============================")
    print("Model:", name)
    print("Accuracy:", round(acc, 4))
    print(classification_report(y_test, preds))

    results[name] = acc

print("\ntesting model: Random guessing")
random_preds = np.random.choice(y_test.unique(), size=len(y_test))
print("Random accuracy:", accuracy_score(y_test, random_preds))

# 1. Naive Bayes + BoW
run_experiment(
    "Naive Bayes (BoW)",
    CountVectorizer(),
    MultinomialNB()
)

# 2. Logistic Regression + TF-IDF
run_experiment(
    "Logistic Regression (TF-IDF)",
    TfidfVectorizer(),
    LogisticRegression(max_iter=1000)
)

# 3. Linear SVM + TF-IDF + bigrams
run_experiment(
    "Linear SVM (TF-IDF + Bigrams)",
    TfidfVectorizer(ngram_range=(1, 2)),
    LinearSVC()
)


print("\nFinal Comparison:")
for name in results:
    print(name, "->", round(results[name], 4))

best_model = max(results, key=results.get)
print("\nBest performing model:", best_model)
