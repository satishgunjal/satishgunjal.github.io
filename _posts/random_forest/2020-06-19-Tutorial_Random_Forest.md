---
title: 'Random Forest'
date: 2020-06-19
permalink: /random_forest/
tags:
  - Classification
  - Regression
  - Sklearn
header:
  teaser: https://raw.githubusercontent.com/satishgunjal/images/master/Random_Forest_Header.png
excerpt: Random forest is supervised learning algorithm and can be used to solve classification and regression problems. Unlike decision tree random forest fits multiple decision trees on various sub samples of dataset and make the predictions by averaging the predictions from each tree. 

---

![Random_Forest_Header](https://raw.githubusercontent.com/satishgunjal/images/master/Random_Forest_Header.png)

Random forest is supervised learning algorithm and can be used to solve classification and regression problems. Since decision-tree create only one tree to fit the dataset, it may cause overfitting and model may not generalize well. Unlike decision tree random forest fits multiple decision trees on various sub samples of dataset and make the predictions by averaging the predictions from each tree. Averaging the results from multiple decision trees help to control the overfitting and results in much better prediction accuracy. As you may have noticed, since this algorithm uses multiple trees hence the name 'Random Forest'

This tutorial is part of my 'Beginner Series Tutorials', I would recommend you to please go through [Decision Tree](https://satishgunjal.github.io/decision_tree/) tutorial first for better understanding.

**Note: Source code used in this article is available at this [Kaggle Kernel](https://www.kaggle.com/satishgunjal/tutorial-random-forest#Classification-Problem-Example)**

# Inner Workings Of Random Forest
* Select few random sub sample from given dataset
* Construct a decision tree for every sub sample and predict the result. To know more about 'decision tree' formation please refer [Inner Workings Of Decision Tree](https://satishgunjal.github.io/decision_tree/#inner-workings-of-decision-tree)
* Perform the voting on prediction from each tree
* At the end select the most voted result as final prediction

For more details about how Random forest classifier splits the data, please refer [Criteria To Split The Data](https://satishgunjal.github.io/decision_tree/#criteria-to-split-the-data)

![Random_Forest](https://raw.githubusercontent.com/satishgunjal/images/master/Random_Forest.png)

## How Do Random Forest Handle Missing Data?
* Please refer above diagram where we have training data set of circle, square and triangle of color red, green and blue respectively.
* There are total 27 training examples. Random forest will create three sub sample of 9 training examples each
* Random forest algorithm will create three different decision tree for each sub sample
* Notice that each tree uses different criteria to split the data
* Now it is straight forward analysis for the algorithm to predict the shape of given figure if its shape and color is known.
  Let's check the predictions of each tree for blue color triangle,
  - Tree 1 will predict: triangle
  - Tree 2 will predict: square
  - Tree 2 will predict: triangle
  Since the majority of voting is for triangle final prediction is 'triangle shape'
* Now, lets check predictions for circle with no color defined (color attribute is missing here)
  - Tree 1 will predict: triangle
  - Tree 2 will predict: circle
  - Tree 2 will predict: circle
  Since the majority of voting is for circle final prediction is 'circle shape'
* Please note this is over simplified example, but you get an idea how multiple tree with different split criteria helps to handle missing features 

# Advantages Of Random Forest
* Reduces the model overfitting by averaging the results from multiple decision trees
* High level of accuracy
* Works well in case of missing data
* Repeated model training is not required

# Disadvantages Of Random Forest
* Random forest generates complex models which are difficult to understand and interpret
* More time and computational resources required as compare to Decision Tree
* Predictions are slower than decision tree

# Classification Problem Example
For classification exercise we are going to use sklearns wine recognition dataset. Objective is to classify wines among three categories based on available data. The data is the results of a chemical analysis of wines grown in the same region in Italy by three different cultivators. There are thirteen different measurements taken for different constituents found in the three types of wine.

## Understanding Dataset
* wine.DESCR > Complete description of dataset
* wine.data > Data to learn. Each training set is 13 digit array of features. 
    - Total training examples 178. 
    - Samples per class [59,71,48]
* wine.feature_names > Array of all 13 feature. Features are as below
    - Alcohol
    - Malic acid
    - Ash
    - Alcalinity of ash  
    - Magnesium
    - Total phenols
    - Flavanoids
    - Nonflavanoid phenols
    - Proanthocyanins
    - Color intensity
    - Hue
    - OD280/OD315 of diluted wines
    - Proline
* wine.target > The classification label. For every training set there is one classification label(0, 1, 2). Here 0 for class_0, 1 for class_1 and 2 for class_2
* wine.filename > CSV file name
* wine.target_names > Name of the classes. It’s an array ['class_0', 'class_1', 'class_2']

## Import The Libraries
* pandas: Used for data manipulation and analysis
* numpy : Numpy is the core library for scientific computing in Python. It is used for working with arrays and matrices.
* datasets: Here we are going to use ‘wine’ and ‘boston house prices’ datasets
* model_selection: Here we are going to use model_selection.train_test_split() for splitting the data
* ensemble: Here we are going to use random forest classifier and regressor


```python
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn import ensemble
```

## Load The Dataset


```python
wine = datasets.load_wine()
print('Dataset structure= ', dir(wine))

df = pd.DataFrame(wine.data, columns = wine.feature_names)
df['target'] = wine.target
df['wine_class'] = df.target.apply(lambda x : wine.target_names[x]) # Each value from 'target' is used as index to get corresponding value from 'target_names' 

print('Unique target values=',df['target'].unique())

df.head()
```

    Dataset structure=  ['DESCR', 'data', 'feature_names', 'target', 'target_names']
    Unique target values= [0 1 2]
    




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
      <th>alcohol</th>
      <th>malic_acid</th>
      <th>ash</th>
      <th>alcalinity_of_ash</th>
      <th>magnesium</th>
      <th>total_phenols</th>
      <th>flavanoids</th>
      <th>nonflavanoid_phenols</th>
      <th>proanthocyanins</th>
      <th>color_intensity</th>
      <th>hue</th>
      <th>od280/od315_of_diluted_wines</th>
      <th>proline</th>
      <th>target</th>
      <th>wine_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127.0</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065.0</td>
      <td>0</td>
      <td>class_0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100.0</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050.0</td>
      <td>0</td>
      <td>class_0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101.0</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185.0</td>
      <td>0</td>
      <td>class_0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113.0</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480.0</td>
      <td>0</td>
      <td>class_0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118.0</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735.0</td>
      <td>0</td>
      <td>class_0</td>
    </tr>
  </tbody>
</table>
</div>



Let visualize the feature values for each type of wine


```python
# label = 0 (wine class_0)
df[df.target == 0].head(3)
```




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
      <th>alcohol</th>
      <th>malic_acid</th>
      <th>ash</th>
      <th>alcalinity_of_ash</th>
      <th>magnesium</th>
      <th>total_phenols</th>
      <th>flavanoids</th>
      <th>nonflavanoid_phenols</th>
      <th>proanthocyanins</th>
      <th>color_intensity</th>
      <th>hue</th>
      <th>od280/od315_of_diluted_wines</th>
      <th>proline</th>
      <th>target</th>
      <th>wine_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127.0</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065.0</td>
      <td>0</td>
      <td>class_0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100.0</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050.0</td>
      <td>0</td>
      <td>class_0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101.0</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185.0</td>
      <td>0</td>
      <td>class_0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# label = 1 (wine class_1)
df[df.target == 1].head(3)
```




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
      <th>alcohol</th>
      <th>malic_acid</th>
      <th>ash</th>
      <th>alcalinity_of_ash</th>
      <th>magnesium</th>
      <th>total_phenols</th>
      <th>flavanoids</th>
      <th>nonflavanoid_phenols</th>
      <th>proanthocyanins</th>
      <th>color_intensity</th>
      <th>hue</th>
      <th>od280/od315_of_diluted_wines</th>
      <th>proline</th>
      <th>target</th>
      <th>wine_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>59</th>
      <td>12.37</td>
      <td>0.94</td>
      <td>1.36</td>
      <td>10.6</td>
      <td>88.0</td>
      <td>1.98</td>
      <td>0.57</td>
      <td>0.28</td>
      <td>0.42</td>
      <td>1.95</td>
      <td>1.05</td>
      <td>1.82</td>
      <td>520.0</td>
      <td>1</td>
      <td>class_1</td>
    </tr>
    <tr>
      <th>60</th>
      <td>12.33</td>
      <td>1.10</td>
      <td>2.28</td>
      <td>16.0</td>
      <td>101.0</td>
      <td>2.05</td>
      <td>1.09</td>
      <td>0.63</td>
      <td>0.41</td>
      <td>3.27</td>
      <td>1.25</td>
      <td>1.67</td>
      <td>680.0</td>
      <td>1</td>
      <td>class_1</td>
    </tr>
    <tr>
      <th>61</th>
      <td>12.64</td>
      <td>1.36</td>
      <td>2.02</td>
      <td>16.8</td>
      <td>100.0</td>
      <td>2.02</td>
      <td>1.41</td>
      <td>0.53</td>
      <td>0.62</td>
      <td>5.75</td>
      <td>0.98</td>
      <td>1.59</td>
      <td>450.0</td>
      <td>1</td>
      <td>class_1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# label = 2 (wine class_2)
df[df.target == 2].head(3)
```




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
      <th>alcohol</th>
      <th>malic_acid</th>
      <th>ash</th>
      <th>alcalinity_of_ash</th>
      <th>magnesium</th>
      <th>total_phenols</th>
      <th>flavanoids</th>
      <th>nonflavanoid_phenols</th>
      <th>proanthocyanins</th>
      <th>color_intensity</th>
      <th>hue</th>
      <th>od280/od315_of_diluted_wines</th>
      <th>proline</th>
      <th>target</th>
      <th>wine_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>130</th>
      <td>12.86</td>
      <td>1.35</td>
      <td>2.32</td>
      <td>18.0</td>
      <td>122.0</td>
      <td>1.51</td>
      <td>1.25</td>
      <td>0.21</td>
      <td>0.94</td>
      <td>4.1</td>
      <td>0.76</td>
      <td>1.29</td>
      <td>630.0</td>
      <td>2</td>
      <td>class_2</td>
    </tr>
    <tr>
      <th>131</th>
      <td>12.88</td>
      <td>2.99</td>
      <td>2.40</td>
      <td>20.0</td>
      <td>104.0</td>
      <td>1.30</td>
      <td>1.22</td>
      <td>0.24</td>
      <td>0.83</td>
      <td>5.4</td>
      <td>0.74</td>
      <td>1.42</td>
      <td>530.0</td>
      <td>2</td>
      <td>class_2</td>
    </tr>
    <tr>
      <th>132</th>
      <td>12.81</td>
      <td>2.31</td>
      <td>2.40</td>
      <td>24.0</td>
      <td>98.0</td>
      <td>1.15</td>
      <td>1.09</td>
      <td>0.27</td>
      <td>0.83</td>
      <td>5.7</td>
      <td>0.66</td>
      <td>1.36</td>
      <td>560.0</td>
      <td>2</td>
      <td>class_2</td>
    </tr>
  </tbody>
</table>
</div>



## Build Machine Learning Model


```python
#Lets create feature matrix X  and y labels
X = df[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium','total_phenols', 'flavanoids', 'nonflavanoid_phenols',
       'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']]
y = df[['target']]

print('X shape=', X.shape)
print('y shape=', y.shape)
```

    X shape= (178, 13)
    y shape= (178, 1)
    

## Create Test And Train Dataset
* We will split the dataset, so that we can use one set of data for training the model and one set of data for testing the model
* We will keep 20% of data for testing and 80% of data for training the model
*If you want to learn more about it, please refer [Train Test Split tutorial](https://satishgunjal.com/train_test_split/)


```python
X_train,X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size= 0.2, random_state= 1)
print('X_train dimension= ', X_train.shape)
print('X_test dimension= ', X_test.shape)
print('y_train dimension= ', y_train.shape)
print('y_train dimension= ', y_test.shape)
```

    X_train dimension=  (142, 13)
    X_test dimension=  (36, 13)
    y_train dimension=  (142, 1)
    y_train dimension=  (36, 1)
    

Now lets train the model using Random Forest Classification Algorithm


```python
"""
To obtain a deterministic behaviour during fitting always set value for 'random_state' attribute
Also note that default value of criteria to split the data is 'gini'
"""
rfc = ensemble.RandomForestClassifier(random_state = 1)
rfc.fit(X_train ,y_train.values.ravel()) # Using ravel() to convert column vector y to 1D array 
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=1, verbose=0,
                           warm_start=False)



## Testing The Model
* For testing we are going to use the test data only
* Question: Predict the wine class of 10th and 30th from test data


```python
print('Actual Wine type for 10th test data sample= ', wine.target_names[y_test.iloc[10]][0])
print('Wine type prediction for 10th test data sample= ',wine.target_names[rfc.predict([X_test.iloc[10]])][0])

print('Actual Wine type for 30th test data sample= ', wine.target_names[y_test.iloc[30]][0])
print('Wine type prediction for 30th test data sample= ',wine.target_names[rfc.predict([X_test.iloc[30]])][0])
```

    Actual Wine type for 10th test data sample=  class_0
    Wine type prediction for 10th test data sample=  class_0
    Actual Wine type for 30th test data sample=  class_1
    Wine type prediction for 30th test data sample=  class_1
    

## Model Score
Check the model score using test data


```python
rfc.score(X_test, y_test)
```




    0.9722222222222222



# Regression Problem Example
For regression exercise we are going to use sklearns Boston house prices dataset. Objective is to predict house price based on available data

Note: I have used same dataset for decision tree regressor example, model score was 66%. If you are interested please refer decision tree implementation of this problem at [Kaggle Notebook](https://www.kaggle.com/satishgunjal/tutorial-decision-tree?scriptVersionId=35851049) or at my blog [Blog](https://satishgunjal.com/decision_tree/#regression-problem-example) 

## Understanding the Boston house dataset
* boston.DESCR > Complete description of dataset
* boston.data > Data to learn. There are 13 features, Attribute 14 is the target. Total 506 training sets
    - CRIM     per capita crime rate by town
    - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
    - INDUS    proportion of non-retail business acres per town
    - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    - NOX      nitric oxides concentration (parts per 10 million)
    - RM       average number of rooms per dwelling
    - AGE      proportion of owner-occupied units built prior to 1940
    - DIS      weighted distances to five Boston employment centers
    - RAD      index of accessibility to radial highways
    - TAX      full-value property-tax rate per USD 10,000
    - PTRATIO  pupil-teacher ratio by town
    - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    - LSTAT    % lower status of the population
    - MEDV     Median value of owner-occupied homes in USD 1000's
* boston.feature_names > Array of all 13 features ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
 'B' 'LSTAT']
* boston.filename > CSV file name
* boston.target > The price valueis in $1000’s

From above details its clear that X = 'boston.data' and y= 'boston.target'

## Lod The Data


```python
boston = datasets.load_boston()
print('Dataset structure= ', dir(boston))

df = pd.DataFrame(boston.data, columns = boston.feature_names)
df['target'] = boston.target

df.head()
```

    Dataset structure=  ['DESCR', 'data', 'feature_names', 'filename', 'target']
    




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
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>



## Build Machine Learning Model


```python
#Lets create feature matrix X  and y labels
X = df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
y = df[['target']]

print('X shape=', X.shape)
print('y shape=', y.shape)
```

    X shape= (506, 13)
    y shape= (506, 1)
    

## Create Test And Train Dataset
* We will split the dataset, so that we can use one set of data for training the model and one set of data for testing the model
* We will keep 20% of data for testing and 80% of data for training the model
*If you want to learn more about it, please refer [Train Test Split tutorial](https://satishgunjal.com/train_test_split/)


```python
X_train,X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size= 0.2, random_state= 1)
print('X_train dimension= ', X_train.shape)
print('X_test dimension= ', X_test.shape)
print('y_train dimension= ', y_train.shape)
print('y_train dimension= ', y_test.shape)
```

    X_train dimension=  (404, 13)
    X_test dimension=  (102, 13)
    y_train dimension=  (404, 1)
    y_train dimension=  (102, 1)
    

Now lets train the model using Random Forest Regressor


```python
"""
To obtain a deterministic behaviour during fitting always set value for 'random_state' attribute
Also note that default value of criteria to split the data is 'mse' (mean squared error)
"""
rfr = ensemble.RandomForestRegressor(random_state= 1)
rfr.fit(X_train ,y_train.values.ravel())  # Using ravel() to convert column vector y to 1D array 
```




    RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                          max_depth=None, max_features='auto', max_leaf_nodes=None,
                          max_samples=None, min_impurity_decrease=0.0,
                          min_impurity_split=None, min_samples_leaf=1,
                          min_samples_split=2, min_weight_fraction_leaf=0.0,
                          n_estimators=100, n_jobs=None, oob_score=False,
                          random_state=1, verbose=0, warm_start=False)



## Testing The Model
* For testing we are going to use the test data only
* Question: Predict the values for every test set in test data


```python
prediction = pd.DataFrame(rfr.predict(X_test), columns = ['prediction'])
# If you notice X_test index starts from 307, so we must reset the index so that we can combine it with prediction values
target = y_test.reset_index(drop=True) # dropping the original index column
target_vs_prediction = pd.concat([target,prediction],axis =1)
target_vs_prediction.T
```




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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>...</th>
      <th>62</th>
      <th>63</th>
      <th>64</th>
      <th>65</th>
      <th>66</th>
      <th>67</th>
      <th>68</th>
      <th>69</th>
      <th>70</th>
      <th>71</th>
      <th>72</th>
      <th>73</th>
      <th>74</th>
      <th>75</th>
      <th>76</th>
      <th>77</th>
      <th>78</th>
      <th>79</th>
      <th>80</th>
      <th>81</th>
      <th>82</th>
      <th>83</th>
      <th>84</th>
      <th>85</th>
      <th>86</th>
      <th>87</th>
      <th>88</th>
      <th>89</th>
      <th>90</th>
      <th>91</th>
      <th>92</th>
      <th>93</th>
      <th>94</th>
      <th>95</th>
      <th>96</th>
      <th>97</th>
      <th>98</th>
      <th>99</th>
      <th>100</th>
      <th>101</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>target</th>
      <td>28.200</td>
      <td>23.900</td>
      <td>16.60</td>
      <td>22.00</td>
      <td>20.800</td>
      <td>23.000</td>
      <td>27.900</td>
      <td>14.500</td>
      <td>21.500</td>
      <td>22.600</td>
      <td>23.700</td>
      <td>31.200</td>
      <td>19.300</td>
      <td>19.400</td>
      <td>19.400</td>
      <td>27.900</td>
      <td>13.900</td>
      <td>50.000</td>
      <td>24.100</td>
      <td>14.600</td>
      <td>16.200</td>
      <td>15.600</td>
      <td>23.800</td>
      <td>25.00</td>
      <td>23.50</td>
      <td>8.300</td>
      <td>13.50</td>
      <td>17.500</td>
      <td>43.100</td>
      <td>11.500</td>
      <td>24.100</td>
      <td>18.500</td>
      <td>50.000</td>
      <td>12.60</td>
      <td>19.800</td>
      <td>24.500</td>
      <td>14.900</td>
      <td>36.20</td>
      <td>11.900</td>
      <td>19.100</td>
      <td>...</td>
      <td>8.500</td>
      <td>14.500</td>
      <td>23.700</td>
      <td>37.200</td>
      <td>41.700</td>
      <td>16.500</td>
      <td>21.700</td>
      <td>22.700</td>
      <td>23.000</td>
      <td>10.500</td>
      <td>21.900</td>
      <td>21.000</td>
      <td>20.400</td>
      <td>21.800</td>
      <td>50.000</td>
      <td>22.000</td>
      <td>23.300</td>
      <td>37.300</td>
      <td>18.000</td>
      <td>19.200</td>
      <td>34.900</td>
      <td>13.400</td>
      <td>22.900</td>
      <td>22.500</td>
      <td>13.000</td>
      <td>24.600</td>
      <td>18.300</td>
      <td>18.100</td>
      <td>23.900</td>
      <td>50.000</td>
      <td>13.600</td>
      <td>22.900</td>
      <td>10.900</td>
      <td>18.900</td>
      <td>22.400</td>
      <td>22.900</td>
      <td>44.800</td>
      <td>21.700</td>
      <td>10.200</td>
      <td>15.400</td>
    </tr>
    <tr>
      <th>prediction</th>
      <td>30.016</td>
      <td>27.473</td>
      <td>20.03</td>
      <td>20.43</td>
      <td>19.754</td>
      <td>19.652</td>
      <td>27.466</td>
      <td>19.151</td>
      <td>20.272</td>
      <td>23.288</td>
      <td>29.018</td>
      <td>30.391</td>
      <td>20.428</td>
      <td>20.396</td>
      <td>20.479</td>
      <td>24.365</td>
      <td>12.379</td>
      <td>40.837</td>
      <td>24.253</td>
      <td>14.179</td>
      <td>19.939</td>
      <td>15.854</td>
      <td>24.264</td>
      <td>23.91</td>
      <td>25.61</td>
      <td>9.537</td>
      <td>14.58</td>
      <td>19.781</td>
      <td>43.808</td>
      <td>12.196</td>
      <td>26.003</td>
      <td>19.588</td>
      <td>47.472</td>
      <td>16.14</td>
      <td>23.495</td>
      <td>20.884</td>
      <td>15.468</td>
      <td>33.67</td>
      <td>13.158</td>
      <td>20.028</td>
      <td>...</td>
      <td>13.442</td>
      <td>14.908</td>
      <td>18.546</td>
      <td>32.699</td>
      <td>42.099</td>
      <td>24.853</td>
      <td>21.545</td>
      <td>20.184</td>
      <td>24.099</td>
      <td>6.958</td>
      <td>18.581</td>
      <td>21.529</td>
      <td>19.587</td>
      <td>20.225</td>
      <td>43.111</td>
      <td>24.424</td>
      <td>27.855</td>
      <td>33.007</td>
      <td>17.175</td>
      <td>20.587</td>
      <td>34.114</td>
      <td>11.567</td>
      <td>24.242</td>
      <td>25.716</td>
      <td>15.393</td>
      <td>24.697</td>
      <td>19.883</td>
      <td>17.703</td>
      <td>28.866</td>
      <td>44.604</td>
      <td>16.316</td>
      <td>21.185</td>
      <td>14.774</td>
      <td>20.524</td>
      <td>23.963</td>
      <td>23.674</td>
      <td>42.712</td>
      <td>20.786</td>
      <td>15.863</td>
      <td>15.956</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 102 columns</p>
</div>



## Model Score
Check the model score using test data


```python
rfr.score(X_test, y_test)
```




    0.90948626473857



**Note that for same dataset with decision tree algorithm, score was around 66% and now with Random Forest algorithm its 91%**
