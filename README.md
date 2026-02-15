# Sports vs Politics Text Classification

**Name:** Vamsi Krishna Reddy  
**Roll Number:** B23CM1045  

---

## Project Overview

This project implements a binary text classification system that categorizes news articles into either **Sport** or **Politics**.

The objective was to compare multiple machine learning techniques using different feature representation methods and analyze their performance. A random prediction baseline was also implemented to verify that the models were genuinely learning patterns and not benefiting from data leakage.

---

## Dataset

The dataset used is the BBC News dataset. Only two categories were selected:

- Sport  
- Politics  

### Dataset Statistics

- Total samples: 928  
- Sport articles: 511  
- Politics articles: 417  

The dataset is reasonably balanced.

An 80–20 stratified split was performed:

- Training set: 742  
- Test set: 186  

Test distribution:

- Sport: 102  
- Politics: 84  

Stratified sampling ensured that both classes were proportionally represented in both training and testing sets.

---

## Preprocessing

Minimal preprocessing was applied:

- Converted all text to lowercase  
- Removed numbers and punctuation using regular expressions  
- No stemming or lemmatization was performed  

The focus was mainly on feature engineering and model comparison.

---

## Feature Representation Methods

Three feature extraction approaches were used:

1. **Bag of Words (BoW)**  
2. **TF-IDF**  
3. **TF-IDF with Bigrams**

Bag of Words captures raw word frequency, TF-IDF emphasizes informative terms, and bigrams help capture short phrase-level patterns.

---

## Models Compared

| Model | Feature Type | Accuracy |
|--------|--------------|----------|
| Naive Bayes | Bag of Words | 1.0000 |
| Logistic Regression | TF-IDF | 0.9892 |
| Linear SVM | TF-IDF + Bigrams | 1.0000 |

---

## Random Baseline (Sanity Check)

Since near-perfect accuracy was observed, a random prediction baseline was implemented to validate model correctness.

Random accuracy obtained:

**0.5215 (~52%)**

This confirms that:

- The dataset is not trivially predictable  
- The trained models significantly outperform random guessing  
- There is no data leakage in the pipeline  

---

## Observations

- The dataset appears highly separable due to strong domain-specific vocabulary.
- Sports articles commonly include words such as: *match, goal, team, player, league*.
- Political articles frequently include: *government, election, minister, parliament, policy*.
- The limited vocabulary overlap makes classification easier.
- Even simple models such as Naive Bayes perform extremely well.
- Including bigrams helps capture phrase-level distinctions.

---

## How to Run

1. Install required libraries:

```
pip install pandas numpy scikit-learn matplotlib
```

2. Place the dataset file (`bbc_data.csv`) in the project directory.

3. Run:

```
python sports_politics.py
```

The script will:

- Train three models  
- Print accuracy and classification reports  
- Display confusion matrices  
- Show final performance comparison  

---

## Project Structure

```
├── sports_politics.py
├── bbc_data.csv
├── README.md
└── report.pdf
```

---

## Limitations

- The dataset is clean and well-separated; real-world data may contain noise and overlap.
- The models rely heavily on vocabulary differences.
- No cross-dataset validation was performed.
- The system does not capture deeper semantic meaning.

---

## Conclusion

This project demonstrates that classical machine learning techniques such as Naive Bayes, Logistic Regression, and Linear SVM are highly effective for structured text classification when the dataset is clearly separable.

The random baseline verification strengthens the credibility of the results and confirms that the models are genuinely learning meaningful patterns.
