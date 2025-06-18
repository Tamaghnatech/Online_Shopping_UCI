# Online Shoppers Intention Prediction
![Online_Shopping](onlineshop.png)
Welcome to this repository, where we perform exploratory analysis, machine learning modeling, threshold optimization, and feature importance analysis on the Online Shoppers Intention dataset. The goal is to predict whether a user session ends in revenue (purchase) or not.

---

## 🧠 Problem Statement

The dataset contains anonymized session-level data from an e-commerce website. Each row represents a session, and the objective is to **predict the `Revenue` column (True or False)** based on a variety of numerical and categorical features such as page durations, traffic types, bounce rates, exit rates, and visitor types.

### ❗ Why this matters

Only **15.4% of all sessions** actually ended in a purchase. That means out of 12,330 online shopping visits, only **1,904 resulted in actual revenue**. The challenge lies in identifying and learning from this highly imbalanced subset. A model that naively predicts "no purchase" would already be 84.5% accurate — so we need a much more intelligent approach to find the real buyers.

---

## 🔄 Overall Analysis Flow

### 🔹 Step 1: Data Loading and Preprocessing

* Loaded the dataset and verified data types and null values
* Converted categorical variables using `get_dummies`
* Split the dataset into training and test sets (80/20)

### 🔹 Step 2: Baseline Modeling

* Used `DummyClassifier(strategy='most_frequent')`
* Achieved **baseline accuracy \~84.5%**, which already reflects the class imbalance

### 🔹 Step 3: Logistic Regression

* Trained a logistic regression model
* Got ROC AUC of **0.88** but with **poor recall (36%)** on minority class

### 🔹 Step 4: Advanced Models

Trained and evaluated the following:

| Model                 | Precision (True) | Recall (True) | F1-score (True) | ROC-AUC |
| --------------------- | ---------------- | ------------- | --------------- | ------- |
| **Random Forest**     | 0.73             | 0.54          | 0.62            | 0.9179  |
| **Gradient Boosting** | 0.73             | 0.59          | 0.65            | 0.9279  |
| **XGBoost**           | 0.67             | 0.60          | 0.64            | 0.9166  |
| **LightGBM**          | 0.70             | 0.58          | 0.64            | 0.9267  |

**Conclusion:** Gradient Boosting showed best F1-ROC tradeoff.

---

## 🔍 Precision-Recall Tradeoff

### 🔸 Precision-Recall Curve

![Precision-Recall Curve](download%20\(83\).png)

### 🔸 F1 vs Threshold

![F1 Score vs Threshold](download%20\(86\).png)

* Explored the impact of threshold tuning.
* Calculated the **best threshold = 0.299** based on F1 score.

---

## ✅ Confusion Matrix at Optimal Threshold

![Confusion Matrix](download%20\(87\).png)

```
Confusion Matrix:
[[1920  164]   # Not Revenue (TN / FP)
 [ 106  276]]  # Revenue (FN / TP)
```

### 📊 Classification Report @ Threshold = 0.299

```
Precision (Revenue): 0.63
Recall (Revenue):    0.72
F1-score:            0.67
```

---

## 🔥 Feature Importance (Gradient Boosting)

### 🔸 Ranked Feature Importance (Sorted)

![Sorted Feature Importances](download%20\(85\).png)

### 🔸 Raw Feature Importance Plot

![Raw Importance](download%20\(84\).png)

**Top features contributing to conversion:**

1. `PageValues` (very dominant)
2. `ProductRelated_Duration`
3. `BounceRates`
4. `ExitRates`
5. `Administrative`, `Informational_Duration`, `Month_Nov`

---

## 🧪 Precision, Recall & F1 vs Threshold (All in One)

![All Metrics vs Threshold](download%20\(88\).png)

* Useful for visualizing tradeoff and picking optimal threshold based on use case.

---

## 📁 Repository Contents

```
├── online.csv              # Dataset
├── model_training.ipynb     # Main Colab notebook with all analysis
├── README.md               # You are here
├── download (83).png      # Precision-Recall Curve
├── download (84).png      # Feature Importances (Raw)
├── download (85).png      # Feature Importances (Sorted)
├── download (86).png      # F1 Score vs Threshold
├── download (87).png      # Confusion Matrix
└── download (88).png      # All Metrics vs Threshold
```

---

## 🙌 Summary

This project walks through a complete **classification pipeline**:

* From baseline model to advanced ensemble classifiers
* From naive threshold to **F1-optimized thresholding**
* Visual analysis of **metrics**, **tradeoffs**, and **feature impact**

The final model, a tuned **Gradient Boosting Classifier**, provides strong **precision-recall tradeoff** at threshold 0.299.

---

Feel free to fork, star, or suggest improvements 🤘

> Made with ❤️ in Colab by Tamaghna Nag
