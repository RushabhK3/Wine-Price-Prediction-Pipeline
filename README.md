# Wine Price Prediction Pipeline

### **Project Summary**

This repository contains a comprehensive Machine Learning pipeline designed to predict the market price of wines using various physicochemical, geographical, and historical features. The project focuses on data cleaning, feature engineering, hyperparameter tuning, and robust model comparison to develop an accurate predictive tool for the wine industry.

The final pipeline utilizes a **Random Forest Regressor** model, which achieved a high degree of predictive accuracy, making it a reliable tool for market analysis and valuation.

---

### **Project Goal**

The primary objective of this project was to build and evaluate a robust regression model capable of accurately estimating the price of a bottle of wine based on a dataset of attributes including winery, vintage, region, rating, and body/acidity scores.

### **Key Results**

The project successfully compared two state-of-the-art ensemble regression techniques: Random Forest Regression and Gradient Boosting Regression.

| Metric | Model | Score |
| :--- | :--- | :--- |
| **R2 Score** | **Random Forest Regressor** | **~80%** |
| Accuracy | Random Forest Regressor | ~80% |

The **Random Forest Regressor** was selected as the final model due to its superior performance, demonstrating an estimated 80% accuracy in predicting unseen wine prices, confirmed by the R2 score.

---

### **Methodology & Pipeline Overview**

The core machine learning workflow is orchestrated by the `Pipeline.py` script. The process adheres to best practices in data science, including:

1.  **Data Loading (`load_data`)**: Ingesting the `wines_SPA.csv` dataset.
2.  **Data Cleaning & Preprocessing (`clean_data`)**: Handling missing values, encoding categorical features (e.g., region and wine type), and standardizing numerical data.
3.  **Feature Selection (`select_features`)**: Using the **SelectKBest** method with the ANOVA F-test to identify and select the most influential features for price prediction.
4.  **Model Training & Tuning**: Splitting the data and using **GridSearchCV** to perform hyperparameter tuning on both the Random Forest and Gradient Boosting Regressors to find the optimal model configuration.
5.  **Evaluation (`evaluate_model`)**: Assessing model performance using standard metrics (MSE, RMSE, R2 Score, and MAPE).

---

### **Repository Structure**

| File | Description |
| :--- | :--- |
| `Pipeline.py` | Contains the complete, modularized Machine Learning pipeline from data loading to model evaluation. |
| `Wine_cli.py` | A lightweight command-line interface (CLI) script for executing the full prediction pipeline with a single command. |
| `wines_SPA.csv` | The raw dataset of wine attributes and prices used for training and testing. |
| `Documentation.pdf` | The **full project report** detailing the problem statement, methodology, model comparisons, and comprehensive results. |

### **How to Run the Project (For Review)**

To quickly execute the pipeline and demonstrate its functionality, you can use the CLI script.

**Prerequisites:**
* Python 3.x
* The required Python libraries (e.g., pandas, numpy, scikit-learn).

**1. Clone the repository and install dependencies:**

```bash

git clone https://github.com/RushabhK3/Wine-Price-Prediction-Pipeline.git 
cd Wine-Price-Prediction-Pipeline
pip install -r requirements.txt
