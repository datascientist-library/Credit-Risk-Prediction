# Credit Risk Prediction

Financial institutions need a risk assessment model to predict whether a borrower will default on a loan. The following dataset contains historic records having details of employee age, status, education level, income, loan amount, credit score and so on. This project aims to predict whether a person will default on a loan using historical credit-related data. It involves extensive exploratory data analysis, feature engineering, and the implementation of various machine learning models and its evaluation.

---

## Problem Statement

The goal is to classify whether a loan applicant will **default (1)** (high risk) or **not default (0)** (low risk) based on input features such as income, employment status, education, and credit history.

---

## Dataset Overview

- **Source**: Provided dataset (`Credit Risk Prediction.csv`)
- **Records**: 100,000 rows
- **Target Variable**: `default(y)` — Binary (0: No Default, 1: Default)
- **Feature Categories**: Numeric + Categorical (e.g., income, age, home ownership, etc.)
1. Personal & Demographic – Age, employment status, education level, marital status. 
2. Financial Information – Income, loan amount, loan term, interest rate, savings. 
3. Credit History – Credit score, credit history length, number of credit lines, late payments, bankruptcies.

---

## Tools & Libraries

- **Programming Language**: Python
- **Pandas, NumPy** – Data manipulation
- **Matplotlib, Seaborn** – Visualization
- **Scikit-learn** – ML models and evaluation
- **XGBoost** – for gradient boosting
- **GridSearchCV** – Hyperparameter tuning

---

## Models Implemented

| Model                | Accuracy | Precision | Recall | F1-score |
|---------------------|----------|----------|----------|----------|
| Logistic Regression        | 51%      | 51%   |   51%   |   51%      |
| Decision Tree              | 66%     | 64%   |   73%   |   68%      |
| Random Forest  | 74%     | 82%   |   61%   |   70%      |
| Gradient Boosting            | 52%     | 52%   |   48%   |   50%      |

---

## Exploratory Data Analysis (EDA)

- Class balance check and visualization
- Correlation heatmaps
- Numerical feature distribution (boxplots)
- Categorical distribution vs. target (count plots, crosstab %)

---

## Key Insights
- The correlation heatmap reveals that the numerical features in the dataset have very weak or near-zero correlations with each other, indicating minimal linear relationships.
- Education level and home ownership shows very limited variation in predicting default.
- Marital status has negligible predictive power regarding defaults.
- Defaulters have lower incomes and higher loan amounts as compared to non-defaulters.

---

## Performance Evaluation

- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- Cross-validation scores
- GridSearch parameter tuning

---

## Project Structure
```
credit_risk_rediction/
├── Credit_Risk_Prediction.csv
├── Credit_Risk_Prediction.ipynb
├── README.md	
└── random_forest_credit_risk_model.ipynb
```

---

## Conclusion

- **Random Forest** performed better than other models with accuracy of **74%**.
- Optimized the model with Hyperparameter tuning and Cross-Validation and it slightly improved the performance of the model.
- With the help of this model, we can predict whether a borrower will be able to repay loans or not by analayzing the features such as 'credit_score', 'loan_amount', 'debt_to_income_ratio', 'late_payments', and 'income'.

---

## Future Work

- Integrate real-time or recent data
- Build an interactive dashboard for live predictions

---

## Author

**Mihir Patil**  
Data Science Capstone Project | MIT World Peace University
