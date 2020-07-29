---
title: 'Hyperparameter Tuning'
date: 2020-07-29
permalink: /hyperparameter/
tags:
  - Hyperparameter
  - Sklearn
header:
  teaser: https://raw.githubusercontent.com/satishgunjal/images/master/Hyperparameter_Tuning_Header_1000x690.png
excerpt: Hyperparameters needs to be specified before fitting the model. Since these parameters can't be estimated from data, getting their value correct have the biggest impact on the model performance. Process to find the optimal hyperparamters is called as hyperparameter tuning.

---

![Hyperparameter_Tuning_Header_1000x690](https://raw.githubusercontent.com/satishgunjal/images/master/Hyperparameter_Tuning_Header_1000x690.png)

# Index
* [Introduction](#1)
* [Parameter vs Hyperparameter](#2)
  - [Parameter](#3)
  - [Hyperparameter](#4)
* [Hypertuning Steps](#5)
* [Tuning Strategy](#6)
  - [Grid Search](#7)
  - [Random Search](#8)
* [Grid Search Example](#9)
  - [Import Libraries](#10)
  - [Load Dataset](#11)
  - [Understanding the Data](#12)
  - [Model Score Without Hyperparameter Tuning](#13)
  - [Model Score Using Hyperparameter Tuning](#14)  
* [Random Search Example](#9)
* [Conclusion](#16)

# Introduction <a id ="1"></a>
As of now we have covered most of the basic machine learning algorithms and you must have noticed that in case of [K-Means Clustering](https://www.kaggle.com/satishgunjal/tutorial-k-means-clustering/notebook) we have to provide the value of 'K', similarly in case of [Support Vector Machines](https://www.kaggle.com/satishgunjal/tutorial-support-vector-machines) we have to choose gamma and Regularization parameter(C). Such parameters which needs to be specified before fitting the model are called as Hyperparameter. Since parameters can't be estimated from data, getting their value correct have the biggest impact on the model performance. Only way to find the best possible value of hyperparameters is to try them and choose the best performing one, this process is knows as hyperparameter tuning. 

# Parameter vs Hyperparameter <a id ="2"></a>

## Parameter <a id ="3"></a>

* Parameter also known as model parameter is a configuration variable which is internal to model and whose value can be estimated from the data
* They are required by the model when making predictions
* They are estimated or learned from data
* They are often not set manually by the practitioner
* They are often saved as part of the learned model
* Some examples of model parameters include:
  * The weights in an artificial neural network
  * The support vectors in a support vector machine
  * The coefficients in a linear regression or logistic regression

## Hyperparameter <a id ="4"></a>

* Hyperparameter are external to the model and whose values cannot be estimated based on the data
* They are often specified by the practitioner (By testing the model with test data)
* They are often tuned for a given predictive modeling problem
* They can often be set using heuristics
* Some examples of model hyperparameters include:
  * The learning rate for training a neural network
  * The gamma and Regularization parameter(C) hyperparameters for support vector machines
  * The K in K-nearest neighbors
  * No of trees (n_estimators) in Random Forest Algorithm
  
![Hyperparameter_Tuning](https://raw.githubusercontent.com/satishgunjal/images/master/Hyperparameter_Tuning1.png)
  
# Hypertuning Steps <a id ="5"></a>

* Make a list of different hyperparameters based on the problem in hand. If there are more than one hyperparameter then make grid with different combination of parameters
* Fit all of them separately to the model. If you have large number of hyperparameters then training time and computational cost will be very high. To reduce it you may try few random combination of hyperparameters, instead of going for every possible permutation.
* Note down the model performance
* Choose the best performing one

Always use cross validation technique for hyperparameter tuning to avoid the model overfitting on test data.

# Tuning Strategy <a id ="6"></a>

Models can have many hyperparameters and to try every permutation of it can be treated as a search problem. Below are the two most common ways to perform the hyperparameter tuning.

## Grid Search <a id ="7"></a>

* Grid search exhaustively considers all parameter combinations for an estimator.
* GridSearchCV implements a “fit” and a “score” method. It also implements “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used.
* The parameters of the estimator used to apply these methods are optimized by cross-validated grid-search over a parameter grid.
* Since grid search goes through every possible combination of hyperparameters, it is computationally expensive and takes longer time to complete.

## Random Search <a id ="8"></a>

* Unlike grid search, random search perform randomized search on hyper parameters and selects few parameter combinations for an estimator.
* RandomizedSearchCV implements a “fit” and a “score” method. It also implements “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used.
* The parameters of the estimator used to apply these methods are optimized by cross-validated search over parameter settings.
* Since random search goes through selective combination of hyperparameters, it is computationally less expensive and takes less time to complete.

# Grid Search Example <a id ="9"></a>

We are going to use [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/overview) competition data. We will to convert this dataset into toy dataset so that we can straightaway jump into hyperparameter tuning

![Titanic_Machine_Learning_from_Disaster](https://raw.githubusercontent.com/satishgunjal/images/master/Titanic_Machine_Learning_from_Disaster.png)

## Import Libraries <a id ="10"></a>

* pandas: Used for data manipulation and analysis
* train_test_split: Sklearn library to split arrays or matrices into random train and test subsets
* GridSearchCV: Sklearn library to perform exhaustive search over specified parameter values for an estimator
* RandomizedSearchCV: Sklearn library to perform randomized search on hyper parameters 
* svm: Sklearn Support Vector Machines library


```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn import svm
```

## Load Dataset <a id ="11"></a>
We will load the dataset into pandas dataframe and convert it into a toy dataset by removing categorical columns and rows and columns with null values.


```python
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

# Remove rows with missing target values
train_data.dropna(axis=0, subset=['Survived'], inplace=True)
y = train_data.Survived # Target variable             
train_data.drop(['Survived'], axis=1, inplace=True) # Removing target variable from training data

train_data.drop(['Age'], axis=1, inplace=True) # Remove columns with null values

# Select numeric columns only
numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
X = train_data[numeric_cols].copy()

print("Shape of input data: {} and shape of target variable: {}".format(X.shape, y.shape))

# X.head() 
pd.concat([X,y], axis=1).head()# Show first 5 training examples
```

    Shape of input data: (891, 5) and shape of target variable: (891,)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Understanding the Data <a id ="12"></a>
Final dataset contains 5 features and 891 training examples. We have to predict which passengers survived the Titanic shipwreck based on available training data. Features that we are going to use in this example are passenger id, ticket class, sibling/spouse aboard, parent/children aboard and ticket fare.

## Model Score Without Hyperparameter Tuning <a id ="13"></a>

We will split the dataset using **train_test_split()** method and use training set for model training and test set for model testing. Later we will use hyperparameter tuning to improve the model performance.


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)
print('X_train dimension= ', X_train.shape)
print('X_test dimension= ', X_test.shape)
print('y_train dimension= ', y_train.shape)
print('y_train dimension= ', y_test.shape)
```

    X_train dimension=  (712, 5)
    X_test dimension=  (179, 5)
    y_train dimension=  (712,)
    y_train dimension=  (179,)
    

Now lets train using SVM classifier. Note that we are using default parameters.


```python
clf= svm.SVC()
clf.fit(X_train, y_train)
print('Model score using default parameters is = ', clf.score(X_test, y_test))
```

    Model score using default parameters is =  0.5977653631284916
    

## Model Score Using Hyperparameter Tuning <a id ="14"></a>
So without hyperparameter tuning we get only 60% accuracy, lets see the model performance using hyperparameter tuning


```python
# Let create parameter grid for GridSearchCV
parameters = {  'C':[0.01, 1, 5],
                'kernel':('linear', 'rbf'),
                'gamma' :('scale', 'auto')
             }
gsc = GridSearchCV(estimator = svm.SVC(), param_grid= parameters,cv= 5,verbose =1)

# Fitting the model for grid search. It will first find the best parameter combination using cross validation. 
# Once it has the best combination, it runs fit again on all data passed to fit (without cross-validation), 
# to built a single new model using the best parameter setting.
gsc.fit(X_train, y_train) 
```

    Fitting 5 folds for each of 12 candidates, totalling 60 fits
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done  60 out of  60 | elapsed: 11.3min finished
    




    GridSearchCV(cv=5, estimator=SVC(),
                 param_grid={'C': [0.01, 1, 5], 'gamma': ('scale', 'auto'),
                             'kernel': ('linear', 'rbf')},
                 verbose=1)



That took long time to complete!!
Now lets review the results from GridSearchCV.


```python
print(f'Best hyperparameters: {gsc.best_params_}') 
print(f'Best score: {gsc.best_score_}')
print('Detailed GridSearchCV result is as below')
gsc_result = pd.DataFrame(gsc.cv_results_).sort_values('mean_test_score',ascending= False)
gsc_result[['param_C','param_kernel','param_gamma','mean_test_score']]
```

    Best hyperparameters: {'C': 0.01, 'gamma': 'scale', 'kernel': 'linear'}
    Best score: 0.6825470304343544
    Detailed GridSearchCV result is as below
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>param_C</th>
      <th>param_kernel</th>
      <th>param_gamma</th>
      <th>mean_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.01</td>
      <td>linear</td>
      <td>scale</td>
      <td>0.682547</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.01</td>
      <td>linear</td>
      <td>auto</td>
      <td>0.682547</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>linear</td>
      <td>scale</td>
      <td>0.676923</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>linear</td>
      <td>auto</td>
      <td>0.676923</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5</td>
      <td>rbf</td>
      <td>scale</td>
      <td>0.672737</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5</td>
      <td>linear</td>
      <td>scale</td>
      <td>0.669891</td>
    </tr>
    <tr>
      <th>10</th>
      <td>5</td>
      <td>linear</td>
      <td>auto</td>
      <td>0.669891</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>rbf</td>
      <td>scale</td>
      <td>0.647454</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.01</td>
      <td>rbf</td>
      <td>scale</td>
      <td>0.623599</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.01</td>
      <td>rbf</td>
      <td>auto</td>
      <td>0.623599</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>rbf</td>
      <td>auto</td>
      <td>0.599675</td>
    </tr>
    <tr>
      <th>11</th>
      <td>5</td>
      <td>rbf</td>
      <td>auto</td>
      <td>0.585620</td>
    </tr>
  </tbody>
</table>
</div>



# Random Search Example <a id ="15"></a>
We will use same dataset and same parameter grid for hyperparameter tuning.


```python
# n_iter=5 > Number of parameter settings that are sampled. 
# So instaed of 12 it will randomly search for only 5 combinations for each fold
rsc = RandomizedSearchCV(estimator = svm.SVC(), param_distributions= parameters,cv=5,n_iter = 5,verbose =1)
rsc.fit(X_train, y_train)
```

    Fitting 5 folds for each of 5 candidates, totalling 25 fits
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done  25 out of  25 | elapsed:  2.5min finished
    




    RandomizedSearchCV(cv=5, estimator=SVC(), n_iter=5,
                       param_distributions={'C': [0.01, 1, 5],
                                            'gamma': ('scale', 'auto'),
                                            'kernel': ('linear', 'rbf')},
                       verbose=1)




```python
print(f'Best hyperparameters: {rsc.best_params_}') 
print(f'Best score: {rsc.best_score_}')
print('Detailed RandomizedSearchCV result is as below')
rsc_result = pd.DataFrame(rsc.cv_results_).sort_values('mean_test_score',ascending= False)
rsc_result[['param_C','param_kernel','param_gamma','mean_test_score']]
```

    Best hyperparameters: {'kernel': 'linear', 'gamma': 'scale', 'C': 0.01}
    Best score: 0.6825470304343544
    Detailed RandomizedSearchCV result is as below
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>param_C</th>
      <th>param_kernel</th>
      <th>param_gamma</th>
      <th>mean_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.01</td>
      <td>linear</td>
      <td>scale</td>
      <td>0.682547</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.01</td>
      <td>linear</td>
      <td>auto</td>
      <td>0.682547</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>linear</td>
      <td>scale</td>
      <td>0.676923</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>rbf</td>
      <td>scale</td>
      <td>0.647454</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>rbf</td>
      <td>auto</td>
      <td>0.585620</td>
    </tr>
  </tbody>
</table>
</div>



# Conclusion <a id ="16"></a>
As you can see from above results hyperparameter tuning helps to find the best parameters which can improve the model performance. Since grid search takes more time and is computationally more heavy, it's not suitable for big datasets. Random search is obvious choice for big datasets, but it doesn't guarantee to find the best parameters as it uses only selected samples of the parameters.

# References:
* [What is the Difference Between a Parameter and a Hyperparameter?](https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/)
* [sklearn.model_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
* [sklearn.model_selection.RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
* [SVM Hyperparameter Tuning using GridSearchCV | ML](https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/)
