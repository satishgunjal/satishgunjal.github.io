---
title: 'Train Test Split'
date: 2020-05-27
permalink: /train_test_split/
tags:
  - Training Data
  - Test Data
  - Python
header:
  teaser: https://raw.githubusercontent.com/satishgunjal/images/master/train_test_split_header_640x480.png
excerpt: In this tutorial we are going to study about train, test data split. We will use sklearn library to do the data split.

---

![train_test_split_header_640x480.png](https://raw.githubusercontent.com/satishgunjal/images/master/train_test_split_header_640x480.png)
 
In machine learning we build model based on given data, but to test the performance of the model we also need test data. Technically we can use the same data for model performance testing but the results won't be reliable. Recommended way is to use the different set of data for model training and model performance testing. Datasets used for model training are called as 'Training Datasets' and datasets used for testing are called as 'Test Datasets'
 
 
## Train and Test Datasets
We usually do 80-20 split for training and test datasets. Its is also good practice to randomly sort the data before splitting into two datasets. We are going to use Sklearn library (model_selection.train_test_split) for splitting the datasets.
 
![train_test_split.png](https://raw.githubusercontent.com/satishgunjal/images/master/train_test_split.png)



## Python Code

### Import Libraries
* pandas: Used for data manipulation and analysis.
* train_test_split: Sklearn train_test_split is used to split the dataset
* linear_model: Sklearn linear regression model




```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
```

### Import Dataset


```
df = pd.read_csv('https://raw.githubusercontent.com/satishgunjal/datasets/master/Fish_Weight_Train_Test_Split.csv')
print('Dimension of dataset= ', df.shape)
df.head(5) # Show first 5 training examples
```

    Dimension of dataset=  (42, 3)
    




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
      <th>Weight</th>
      <th>Height</th>
      <th>Width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>242.0</td>
      <td>11.5200</td>
      <td>4.0200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>290.0</td>
      <td>12.4800</td>
      <td>4.3056</td>
    </tr>
    <tr>
      <th>2</th>
      <td>340.0</td>
      <td>12.3778</td>
      <td>4.6961</td>
    </tr>
    <tr>
      <th>3</th>
      <td>363.0</td>
      <td>12.7300</td>
      <td>4.4555</td>
    </tr>
    <tr>
      <th>4</th>
      <td>450.0</td>
      <td>13.6024</td>
      <td>4.9274</td>
    </tr>
  </tbody>
</table>
</div>



### Understanding The Dataset
* There are total 42 rows(training samples) and 4 columns in dataset.
* Features/input values/independent variables are ‘Height’ and ‘Width’
* Labels/Target/output value/dependent variable is ‘Weight’
 
Let's create separate dataframe for features and labels. It is required for splitting the dataset.


```
X = df.drop(['Weight'], axis='columns')
X.head(5)
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
      <th>Height</th>
      <th>Width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11.5200</td>
      <td>4.0200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.4800</td>
      <td>4.3056</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.3778</td>
      <td>4.6961</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.7300</td>
      <td>4.4555</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.6024</td>
      <td>4.9274</td>
    </tr>
  </tbody>
</table>
</div>




```
y = df.Weight
y.head()
```




    0    242.0
    1    290.0
    2    340.0
    3    363.0
    4    450.0
    Name: Weight, dtype: float64



Now we have features and target variables ready, lets split the data into training and test datasets

### Using Sklearn train_test_split Method
* train_test_split() method takes three arguments input features, labels and test_size. 
* Test size determines the percentage of split. e.g. test_size = 0.2, means 80% training data and 20% test data.
* random_state is optional argument.

### What Is random_state
* It is used for initializing the internal random number generator, which will decide the splitting of data into train and test datasets
* Order of the data will be same for a particular value of random_state. For e.g. for 'random_state=1' no matter how many times you run the code you will get same data in training and test split
* You can use any integer value for random_state. Just remember one thing if you don't pass any value, then it will use default value 'None' and split data randomly every time you execute the code.


 



```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

print('X_train dimension= ', X_train.shape)
print('X_test dimension= ', X_test.shape)
print('y_train dimension= ', y_train.shape)
print('y_train dimension= ', y_test.shape)
```

    X_train dimension=  (33, 2)
    X_test dimension=  (9, 2)
    y_train dimension=  (33,)
    y_train dimension=  (9,)
    

Lets visulaize the training and test data using scatter plot


```
import matplotlib.pyplot as plt
plt.scatter(X_train.Height,y_train, color='blue', label='Training Data')
plt.scatter(X_test.Height,y_test, color='orange', label='Test Data')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Training Vs Test Data For Height Feature')
plt.rcParams["figure.figsize"] = (10,6)
plt.legend()
```




![png](https://raw.githubusercontent.com/satishgunjal/images/master/training_vs_test_data_for_height_feature.png)


### Linear Model Training Using Training Dataset
Since we have training and test dataset ready, lets use training dataset for linear model training.


```
lm = linear_model.LinearRegression()
lm.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



### Linear Model Testing Using Test Dataset
Lets use test dataset for linear model performance testing.


```
lm.score(X_test, y_test)
```




    0.7153810385773975



### Linear Model Testing Using Training Dataset
Lets use training dataset for linear model performance testing. Notice the difference in performance score.


```
lm.score(X_train, y_train)
```




    0.8488919680474343



### Never Test On Training Data
* As you can notice score with training data is higher than score with test data.
* Higher score is misleading in this case.
* Model which dont use separate dataset for testing may have higher performance score but it wont generalize well and give misleading predictions with real world data.

Hence forward, in all the tutorials we are going to use training and test dataset for model training and testing.
