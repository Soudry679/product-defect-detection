
# 🏭 Defect Detection in Manufacturing – Machine Learning Project

This project develops a machine learning pipeline to identify defective products in a manufacturing process. The core objective is to minimize false negatives (i.e., failing to catch defective products), which are especially critical in industrial quality control.

---

## 📌 Objective

Predict whether a manufactured product is **defective** (`Yes`/`No`) based on production metrics such as sensor readings, operator experience, and line information.

---

## 📁 Project Structure

- `train_data.csv` – Labeled dataset for training and evaluation  
- `test_data.csv` – Unlabeled data for final predictions  
- `predictions.csv` – Output predictions using the final model  
- `main.py` or `project.ipynb` – Core code, structured into logical blocks  

---

## ⚙️ Tools & Libraries

- Python 3.10+  
- [pandas](https://pandas.pydata.org/)  
- [NumPy](https://numpy.org/)  
- [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/)  
- [Scikit-learn](https://scikit-learn.org/stable/)

---

## 🧪 Pipeline Overview

### 1. **Data Cleaning & Preprocessing**
- Impute missing values (`mean` / `median`)
- Encode categorical variable (`ProductionLine`) using one-hot encoding

### 2. **Feature Selection & Engineering**
- Select top 8 features using `SelectKBest` with ANOVA F-score
- Add polynomial **interaction terms** using `PolynomialFeatures` (no squared terms)

### 3. **Model Training**
- Models trained:
  - Logistic Regression
  - Random Forest (with `GridSearchCV` for hyperparameter tuning)
  - K-Nearest Neighbors
  - AdaBoost

### 4. **Model Evaluation**
- **Primary Metric**: Recall  
- Evaluate using both test split and **Stratified K-Fold cross-validation**  
- Visualize **Precision-Recall vs Threshold** for logistic regression

### 5. **Threshold Tuning**
- Chose threshold = `0.3` to boost recall at the cost of precision
- Calculated confusion matrix, precision, recall, and F1 score for the tuned model

### 6. **Final Prediction**
- Used logistic regression with interaction features and tuned threshold
- Predictions generated for unseen test set and exported as `predictions.csv`

---

## 📊 Results

- **Logistic Regression with interaction features** outperformed other models in recall
- Threshold tuning significantly increased the model's ability to detect defects

---

## 💡 Why Logistic Regression?

Despite testing several models, logistic regression was selected for final deployment due to:
- High recall with interaction terms
- Interpretability
- Stable performance across folds and datasets
- Better generalization than overfit ensemble models in this case

---

## 🔚 Final Notes

This project simulates a real-world classification problem in **smart manufacturing** and **predictive quality assurance**. It highlights the balance between precision and recall when defect detection is mission-critical.

---

## 📝 Author

Aviel Soudry  
Project for Machine Learning Course – 2025  
Tel Aviv University

