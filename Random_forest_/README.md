# Random Forest Classification on Adult Income Dataset

This project demonstrates the use of Random Forests and related machine learning techniques to predict whether a person's income exceeds $50K/year based on census data. The workflow includes data cleaning, feature engineering, model training, hyperparameter tuning, and feature importance analysis.

## Dataset

The dataset used is the [UCI Adult Income dataset](https://archive.ics.uci.edu/ml/datasets/adult), provided in the file `adult.data`. Each row represents an individual, with features such as age, education, occupation, hours-per-week, and more. The target variable is `income`, which is either `<=50K` or `>50K`.

## Project Steps

### 1. Data Loading and Initial Exploration

- The data is loaded using pandas, with column names assigned for clarity.
- The distribution of the target variable (`income`) is displayed to understand class balance.

### 2. Data Cleaning

- Extra whitespace is stripped from all string columns to ensure consistency in categorical values.

### 3. Feature Selection and Engineering

- A subset of relevant features is selected for modeling, including both numerical and categorical variables.
- Categorical variables are converted to dummy/indicator variables using one-hot encoding.
- The target variable is binarized: 0 for `<=50K`, 1 for `>50K`.

### 4. Train-Test Split

- The dataset is split into training and testing sets (80/20 split) to evaluate model performance on unseen data.

### 5. Model Training: Random Forest

- A Random Forest Classifier is trained with default parameters.
- The accuracy on the test set is reported.

### 6. Hyperparameter Tuning

- The `max_depth` parameter of the Random Forest is tuned over a range of values (1-25).
- For each depth, the model is trained and evaluated on both train and test sets.
- The best depth (yielding highest test accuracy) is identified and reported.
- Accuracy trends are visualized with a plot comparing train and test accuracy.

### 7. Feature Importance Analysis

- The best Random Forest model is used to compute feature importances.
- Feature importance visualization:
  - Bar plot showing the importance scores for all features
  - Top 5 most important features are highlighted and analyzed

### 8. Advanced Feature Engineering

- New features are created by binning the `education-num` column into categories:
  - "HS or less"
  - "College to Bachelors"
  - "Masters or more"
- The feature set is updated, and the train-test split is repeated.
- Hyperparameter tuning and feature importance analysis are repeated with the new features.

### 9. Model Evaluation

- Comprehensive performance metrics are calculated:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC Score
- ROC Curve visualization:
  - Plot showing True Positive Rate vs False Positive Rate
  - AUC (Area Under Curve) score displayed
  - Key threshold points analyzed

### 10. Cross-Validation

- 5-fold cross-validation is performed to ensure model stability
- Mean and standard deviation of cross-validation scores are reported
- Model parameters are documented for reproducibility

## Files

- `main.ipynb`: Jupyter notebook containing all code for data processing, modeling, and analysis.
- `adult.data`: Raw dataset file.
- `README.md`: This documentation file.

## How to Run

1. Ensure you have Python 3 and the required libraries installed:
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scikit-learn

2. Open `main.ipynb` in Jupyter Notebook or VS Code.

3. Run all cells in order to reproduce the analysis and results.

## Results

- The notebook reports the best accuracy achieved on the test set for different feature sets and model depths.
- Feature importance rankings help interpret which variables most influence income prediction.
- ROC curve analysis provides insights into model performance across different classification thresholds.
- Cross-validation ensures the model's stability and generalizability.

## Notes

- The project demonstrates the importance of data cleaning and feature engineering in improving model performance.
- Random Forests are robust to overfitting, but tuning parameters like `max_depth` is still important.
- Feature importance analysis provides insights into the drivers of income in the dataset.
- Visualization techniques (ROC curves, feature importance plots) enhance model interpretability.

---

**Author:**  
[Sagar Roy]

**License:**  
For educational use only.
