# House Price Prediction — README

## Project overview

This repository contains a compact machine-learning project that predicts house sale prices using the AMES dataset. The core script (`house_price_prediction.py`) loads a prepared dataset, performs data preparation steps, trains an ElasticNet regression model using grid search, and prints evaluation metrics. Prior data preparation was made to deal with outliers and missing data.

## High-level flow implemented in the code

1. **Data loading**

   - The script reads `AMES_Final_DF.csv` into a pandas DataFrame.
   - The target column is `SalePrice` and the remaining columns are used as features.

2. **Feature / target split**

   - `X` is created by dropping the `SalePrice` column.
   - `y` is `SalePrice`.

3. **Train/test split**

   - A single split is performed with `test_size=0.1` and a fixed `random_state` for reproducibility.

4. **Preprocessing (as implemented in the script)**

   - A `StandardScaler()` instance is created and used in the code path that trains the model. (The script references scaling of features before model fitting.)
   - The dataset name `AMES_Final_DF.csv` indicates preprocessing steps (outlier handling, missing-value treatment, categorical encoding) were applied before the CSV was exported.

5. **Model selection and training**

   - The model family used is `ElasticNet` from scikit-learn.
   - Hyperparameter tuning is performed with `GridSearchCV`. The grid includes `alpha` and `l1_ratio`.
   - `GridSearchCV` is fit on the training split and the best estimator is selected.

6. **Evaluation**

   - The trained model predicts on the held-out test set.
   - Two metrics are computed and printed:

     - RMSE (root mean squared error)
     - MAE (mean absolute error)

## Results

```
Best estimator: ElasticNet(alpha=100, l1_ratio=1)
RMSE on test set: 20619.57686767851
MAE on test set: 14218.352383897884
```

---
