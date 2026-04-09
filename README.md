# Employee Retention Prediction Web App

 **Live App:** https://your-app-name.streamlit.app

---

## Overview

This is an end-to-end machine learning web application built using Streamlit to predict employee retention risk.

Users can upload their dataset, select a target column, and the system automatically trains multiple models to identify the best one and classify employees into risk zones.

---

## Features

* Upload CSV dataset
* Select target column dynamically
* Automatic preprocessing (missing values & encoding)
* Conditional SMOTE for imbalanced data
* Multiple ML models:

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * Gradient Boosting
  * KNN
  * SVM
* Best model selection
* Classification report & confusion matrix
* Employee risk zones:

  * Safe Zone
  * Medium Risk Zone
  * High Risk Zone
* Single employee prediction

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib
* Seaborn
* Imbalanced-learn

---

## How to Run

Install dependencies:
pip install -r requirements.txt

Run app:
streamlit run hr_app.py

---

## Author

Shivam Gupta
