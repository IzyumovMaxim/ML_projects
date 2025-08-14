import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump

df = pd.read_csv("../DATA/AMES_Final_DF.csv")

X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

scaler = StandardScaler()
scaler.fit(X_train)

X_test = scaler.transform(X_test)
X_train = scaler.transform(X_train)

model = ElasticNet()

grid = {
    'alpha': [0.1, 0.5, 0.7, 1, 2, 3, 5, 8, 10, 20, 30, 50, 100],
    'l1_ratio': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
}
search = GridSearchCV(
    estimator=model,
    param_grid=grid,
    scoring = 'neg_mean_squared_error',
    cv = 5,
    verbose=2
)
search.fit(X_train, y_train)
y_pred = search.predict(X_test)

print(search.best_estimator_)
print(np.sqrt(mean_squared_error(y_pred, y_test)))
print(mean_absolute_error(y_pred, y_test))
dump(search.best_estimator_, 'house_price_prediction_model.joblib')