## 使用sklearn构建完整的分类项目

以Iris为例


```python
from sklearn import datasets
import pandas as pd
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature = iris.feature_names
data = pd.DataFrame(X,columns=feature)
data['target'] = y
```


```python
wine = datasets.load_wine()
X_wine = wine.data
y_wine = wine.target
feature = wine.feature_names
data_wine = pd.DataFrame(X_wine,columns=feature)
data_wine['target'] = y_wine
```

## Some additional notes over the confusion matrix:

Precision = TP/(TP + FP), which means the the percentage of correct predictions out of all the positive predicionts

Recall = TP / (TP + FN), which means how much percentage of positive cases are correctly predicted out of all the positive cases.

F1 score = 2 * (precision * recall) / (precision + recall)

F1 score is the harmonic mean of Recall and Precision, the aim of using the harmonic mean is that it would be high only if all the components are of high values.

ROC aims to tune the threshold values of classifiers, it can also work as the method to compare the performances of different classifiers

More info about ROC curve and AUC score:

https://www.youtube.com/watch?v=4jRBRDbJemM

## Some Classic Classification Algorithms:

### Logistic Regression

Self explanative - apply a logistic functino over a linear regression formula

一个小扩展： 逻辑回归不适用于高维高稀疏性的数据，面对这样分布的数据，可以通过施密特正交化后的数据进行建模提高逻辑回归模型的表现

## Naive Bayes

## Support Vector Machine

## Decision Tree

## Neural Networks

are all classic machine learning classification models


```python

```
