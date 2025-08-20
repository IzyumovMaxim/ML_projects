# Sonar data classification


## Описание

Небольшой проект/упражнение по KNN для бинарной классификации (предсказание является объект камнем или миной).

## Что делает проект

* Загружает набор данных.
* Выполняет предобработку (масштабирование).
* Обучает модель `KNN` из `scikit-learn`.
* Оценивает модель метриками: accuracy, precision, recall.

## Структура проекта

```
/ (root)
├─ sonar_data_classification.ipynb
└─ README.md
```

## Зависимости

* Python 3.12+
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* 
### Метрики

```
              precision    recall  f1-score   support

           0       0.77      0.91      0.83        11
           1       0.88      0.70      0.78        10

    accuracy                           0.81        21
   macro avg       0.82      0.80      0.81        21
weighted avg       0.82      0.81      0.81        21
```

Confusion matrix (строки — истинные метки, столбцы — предсказанные):

```
array([[10,  1],
       [ 3,  7]])
```


