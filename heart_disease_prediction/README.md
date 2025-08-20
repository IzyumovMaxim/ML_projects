# Heart desiase classification

## Описание

Небольшой проект/упражнение по логистической регрессии для бинарной классификации (предсказание наличия сердечного заболевания).

## Что делает проект

* Загружает набор данных `DATA/heart.csv`.
* Выполняет предобработку (масштабирование, кодирование при необходимости).
* Обучает модель `LogisticRegression` из `scikit-learn`.
* Оценивает модель метриками: accuracy, precision, recall, ROC AUC, confusion matrix.
* Визуализирует результаты (ROC-кривая, матрица ошибок и др.).

## Структура проекта

```
/ (root)
├─ heart_disease_project.ipynb  
└─ README.md
```

## Зависимости

* Python 3.12+
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
## Результаты
```
              precision    recall  f1-score   support

           0       0.86      0.80      0.83        15
           1       0.82      0.88      0.85        16

    accuracy                           0.84        31
   macro avg       0.84      0.84      0.84        31
weighted avg       0.84      0.84      0.84        31
```
