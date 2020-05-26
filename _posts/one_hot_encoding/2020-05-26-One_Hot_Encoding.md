---
title: 'One Hot Encoding'
date: 2020-05-26
permalink: /one_hot_encoding/
tags:
  - Dummy Variable
  - Dummy Variable Trap
  - Categorical Data
  - Python
excerpt: In this tutorial we are going to study about One Hot Encoding. We will also use pandas and sklearn libraries to convert categorical data into numeric data.
---
![one_hot_encoding_header.png](https://raw.githubusercontent.com/satishgunjal/images/master/one_hot_encoding_header_640x452.png)
 
One of the most important thing while working on applied machine learning is well formatted data. We all know that how messy real world data can be. That is the reason why most of the time is spent on data preprocessing. Most of the machine learning models cannot operate if data is not in numeric format. That's where One Hot Encoding come in picture. In short, it is a technique used to convert categorical text data into numeric format.
 
Note that One Hot Encoding is not a silver bullet that will convert any kind of text data from your dataset to numeric format. Its useful only with Categorical data.
 
## What Is Categorical Data
As name suggest its data which can be divided into categories or groups. Examples of categorical data/variable are sex(male, female, other) and education levels(Graduate, Masters, PhD)
If categorical variables don't have any numeric order or relationship between them then they are called as **Nominal Variables**. For example sex is Nominal Categorical variable.
On other hand if categorical variables have numeric order or relationship between them then they are called as **Ordinal Variables**. For example education level(graduate, Masters, PhD) is Ordinal Categorical variable.
 
## How To Convert Categorical Data To Numeric Data
Now we know that we have to convert categorical data into numeric format so that our model can operate on it. There are two ways we can convert categorical data into numeric format. Label Encoding and One Hot Encoding.
 
## Label Encoding:
  - It is also knows as 'Integer Encoding' because in this technique we simply assign numbers to each category. Numbering starts from 1 and then increase it for each category.
 
  ![label_encoding.png](https://raw.githubusercontent.com/satishgunjal/images/master/label_encoding.png) 
 
### Issue With Label Encoding
 * Label encoding only works with Ordinal variables where each category can be represented with numbers with some kind of order.
 * You have to also make sure to get that order right in order to avoid any prediction errors
 * Consider the example of sex categories where there is no natural order in categories. Machine learning model perform series of mathematical operation on given data in order to establish the relationship between input features. If model calculates the average between category 'male' and 'other' then we get (1+3)/2 = 2 which is same as label value of 'female'. This is just an example you can imagine what will happen to the model when it finds such kind of correlation in data!
 
## One Hot Encoding
* In One Hot Encoding we use Binary Categorizing. We create separate column for each category and assign the binary value 1 or 0 to it.
* It is most commonly used technique to convert categorical data in numeric format.
* Since we create separate column with binary value for each category it avoids any false correlation between unrelated categories
* Extra variables created for each category are called as Dummy Variables
 
  ![binary_encoding.png](https://raw.githubusercontent.com/satishgunjal/images/master/binary_encoding.png) 
 
## Dummy Variable Trap
*  Dummy variable trap occurs when dummy variables are multicolinear with each other. That means one dummy variables value can be predicted using other dummy variables.
* Remember that machine learning model perform series of mathematical operation on given data in order to establish the relationship between input features. And if there is multicolinearity between dummy variables it will affect the model performance.
* Best way to avoid this is to drop one of the dummy variable column.
 



## Python Code
Let's see how to do One Hot Encoding using pandas and sklearn libraries using real world data.



### Import the required libraries
* pandas: Used for data manipulation and analysis. Here we are going to use 
'get_dummies()' method for One Hot Encoding
* numpy : Numpy is the core library for scientific computing in Python. It is used for working with arrays and matrices.
* ColumnTransformer : Sklearn ColumnTransformer is used to apply data transform to different columns of dataset. Here we are using it to apply binary data transform to categorical data column.
* OneHotEncoder : Sklearn OneHotEncoder for binary encoding
* linear_model: Sklearn linear regression model


```
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
```

### Import the dataset


```
df = pd.read_csv('https://raw.githubusercontent.com/satishgunjal/datasets/master/Fish_Weight_One_Hot_Encoding.csv')
print('Dimension of dataset= ', df.shape)
print('Types of spcies= ', df.Species.unique()) # To get unique values from column
df.sample(5) # Display random 5 training examples
```

    Dimension of dataset=  (42, 4)
    Types of spcies=  ['Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt']
    




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
      <th>Species</th>
      <th>Weight</th>
      <th>Height</th>
      <th>Width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24</th>
      <td>Perch</td>
      <td>5.9</td>
      <td>2.1120</td>
      <td>1.4080</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Bream</td>
      <td>242.0</td>
      <td>11.5200</td>
      <td>4.0200</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Parkki</td>
      <td>120.0</td>
      <td>8.3922</td>
      <td>2.9181</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Whitefish</td>
      <td>540.0</td>
      <td>10.7440</td>
      <td>6.5620</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Parkki</td>
      <td>150.0</td>
      <td>8.8928</td>
      <td>3.2928</td>
    </tr>
  </tbody>
</table>
</div>



#### Understanding the dataset
* There are total 42 rows(training samples) and 4 columns in dataset.
* Each column details are as below
  - Species:  Type of fish ('Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt')
  - Weight:   Weight of fish in gram
  - Height: Height in CM
  - Width:  Diagonal width in CM
* Features/input values/independent variables are 'Species', 'Height' and 'Width'
* Target/output value/dependent variable is 'Weight'
 
We can use above data to create a linear model to estimate the weight of the fish based on its measurement values. But since Species data is in text format either we have to drop it or convert it into numeric format.
 
**Fish species is categorical variable. Means we can't use label encoding here. We will use One Hot Encoding to convert fish species types into numeric format. And at the end we will also perform linear regression to test our dataset.**
 



### One Hot Encoding Using Pandas
Pandas get_dummies() method will create separate column for each category and assign binary value to it


```
dummies = pd.get_dummies(df.Species)
dummies
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
      <th>Bream</th>
      <th>Parkki</th>
      <th>Perch</th>
      <th>Pike</th>
      <th>Roach</th>
      <th>Smelt</th>
      <th>Whitefish</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Note above, new dummy variables for each species and their binary values.
 
Let's add newly created dummy variables to existing dataset


```
# pnadas conact method is used to merge two dataframes. 
df1 = pd.concat([df, dummies], axis='columns')
df1.sample(5)
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
      <th>Species</th>
      <th>Weight</th>
      <th>Height</th>
      <th>Width</th>
      <th>Bream</th>
      <th>Parkki</th>
      <th>Perch</th>
      <th>Pike</th>
      <th>Roach</th>
      <th>Smelt</th>
      <th>Whitefish</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23</th>
      <td>Parkki</td>
      <td>140.0</td>
      <td>8.5376</td>
      <td>3.2944</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Bream</td>
      <td>242.0</td>
      <td>11.5200</td>
      <td>4.0200</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Parkki</td>
      <td>60.0</td>
      <td>6.5772</td>
      <td>2.3142</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Pike</td>
      <td>430.0</td>
      <td>7.2900</td>
      <td>4.5765</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bream</td>
      <td>363.0</td>
      <td>12.7300</td>
      <td>4.4555</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Since we have dummy variables for Species feature, we can drop 'Species' column and also to avoid the 'Dummy Variable Trap' we will drop 'Whitefish' column


```
df2 = df1.drop(['Species','Whitefish'], axis='columns')
df2.sample(5)
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
      <th>Weight</th>
      <th>Height</th>
      <th>Width</th>
      <th>Bream</th>
      <th>Parkki</th>
      <th>Perch</th>
      <th>Pike</th>
      <th>Roach</th>
      <th>Smelt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24</th>
      <td>5.9</td>
      <td>2.1120</td>
      <td>1.4080</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>78.0</td>
      <td>5.5756</td>
      <td>2.9044</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1000.0</td>
      <td>12.3540</td>
      <td>6.5250</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>140.0</td>
      <td>8.5376</td>
      <td>3.2944</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>60.0</td>
      <td>6.5772</td>
      <td>2.3142</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Now this is our final dataset. We can use this for linear regression.


```
# Create feature matrix
X = df2.drop(['Weight'],axis = 'columns') 
# Create target vector
y = df2.Weight 

lm = linear_model.LinearRegression()
#Train the model using training data
lm.fit(X,y)
#Check model score
lm.score(X,y) 
# Note: We shouldnt use same dataset to check model score, this is out of scope of this tutorial.
```




    0.9058731241968216



### One Hot Encoding Using Sklearn Preprocessing

We will again start with original dataset 'df'


```
df.head(5)
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
      <th>Species</th>
      <th>Weight</th>
      <th>Height</th>
      <th>Width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bream</td>
      <td>242.0</td>
      <td>11.5200</td>
      <td>4.0200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bream</td>
      <td>290.0</td>
      <td>12.4800</td>
      <td>4.3056</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bream</td>
      <td>340.0</td>
      <td>12.3778</td>
      <td>4.6961</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bream</td>
      <td>363.0</td>
      <td>12.7300</td>
      <td>4.4555</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bream</td>
      <td>450.0</td>
      <td>13.6024</td>
      <td>4.9274</td>
    </tr>
  </tbody>
</table>
</div>




```
# creating one hot encoder object with categorical feature 0 indicating the first column of Species
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])],  remainder='passthrough')

# fit_transform() is combination of '.fit' and '.transform' command. .fit takes Species column and converts everything to numeric data and .tyransform just applies that conversion.
data = np.array(columnTransformer.fit_transform(df), dtype = np.str)
# Creating final dataframe using binary encoded sopecies dummy variables
df1 = pd.DataFrame(data, columns=['Bream','Parkki','Perch','Pike','Roach','Smelt','Whitefish','Weight','Height','Width'])
df1

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
      <th>Bream</th>
      <th>Parkki</th>
      <th>Perch</th>
      <th>Pike</th>
      <th>Roach</th>
      <th>Smelt</th>
      <th>Whitefish</th>
      <th>Weight</th>
      <th>Height</th>
      <th>Width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>242.0</td>
      <td>11.52</td>
      <td>4.02</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>290.0</td>
      <td>12.48</td>
      <td>4.3056</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>340.0</td>
      <td>12.3778</td>
      <td>4.6961</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>363.0</td>
      <td>12.73</td>
      <td>4.4555</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>450.0</td>
      <td>13.6024</td>
      <td>4.9274</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>500.0</td>
      <td>14.1795</td>
      <td>5.2785</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>4.1472</td>
      <td>2.2680000000000002</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>69.0</td>
      <td>5.2983</td>
      <td>2.8217</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>78.0</td>
      <td>5.5756</td>
      <td>2.9044</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>87.0</td>
      <td>5.6166</td>
      <td>3.1746</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>120.0</td>
      <td>6.216</td>
      <td>3.5742</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.4752</td>
      <td>3.3516</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>270.0</td>
      <td>8.3804</td>
      <td>4.2476</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>270.0</td>
      <td>8.1454</td>
      <td>4.2485</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>306.0</td>
      <td>8.777999999999999</td>
      <td>4.6816</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>540.0</td>
      <td>10.744000000000002</td>
      <td>6.562</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>800.0</td>
      <td>11.7612</td>
      <td>6.5736</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1000.0</td>
      <td>12.354000000000001</td>
      <td>6.525</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>55.0</td>
      <td>6.8475</td>
      <td>2.3265</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>60.0</td>
      <td>6.5772</td>
      <td>2.3142</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>90.0</td>
      <td>7.4052</td>
      <td>2.673</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>120.0</td>
      <td>8.3922</td>
      <td>2.9181</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>150.0</td>
      <td>8.8928</td>
      <td>3.2928</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>140.0</td>
      <td>8.5376</td>
      <td>3.2944</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.9</td>
      <td>2.112</td>
      <td>1.4080000000000001</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>32.0</td>
      <td>3.528</td>
      <td>1.9992</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>3.824</td>
      <td>2.432</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>51.5</td>
      <td>4.5924</td>
      <td>2.6316</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>70.0</td>
      <td>4.588</td>
      <td>2.9415</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>5.2224</td>
      <td>3.3216</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>200.0</td>
      <td>5.568</td>
      <td>3.3756</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>300.0</td>
      <td>5.7078</td>
      <td>4.158</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>300.0</td>
      <td>5.9364</td>
      <td>4.3844</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>300.0</td>
      <td>6.2884</td>
      <td>4.0198</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>430.0</td>
      <td>7.29</td>
      <td>4.5765</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>345.0</td>
      <td>6.396</td>
      <td>3.977</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>6.7</td>
      <td>1.7388</td>
      <td>1.0476</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.5</td>
      <td>1.972</td>
      <td>1.16</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>1.7284</td>
      <td>1.1484</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>9.7</td>
      <td>2.1959999999999997</td>
      <td>1.38</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>9.8</td>
      <td>2.0832</td>
      <td>1.2772</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>8.7</td>
      <td>1.9782</td>
      <td>1.2852</td>
    </tr>
  </tbody>
</table>
</div>



Note above, new dummy variables for each species and their binary values.

To avoid the 'Dummy Variable Trap' we will drop 'Whitefish' column


```
df2 = df1.drop(['Whitefish'], axis = 'columns')
df2.sample(10)
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
      <th>Bream</th>
      <th>Parkki</th>
      <th>Perch</th>
      <th>Pike</th>
      <th>Roach</th>
      <th>Smelt</th>
      <th>Weight</th>
      <th>Height</th>
      <th>Width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>5.2224</td>
      <td>3.3216</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>78.0</td>
      <td>5.5756</td>
      <td>2.9044</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>450.0</td>
      <td>13.6024</td>
      <td>4.9274</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>69.0</td>
      <td>5.2983</td>
      <td>2.8217</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>363.0</td>
      <td>12.73</td>
      <td>4.4555</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>120.0</td>
      <td>6.216</td>
      <td>3.5742</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>430.0</td>
      <td>7.29</td>
      <td>4.5765</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>55.0</td>
      <td>6.8475</td>
      <td>2.3265</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>540.0</td>
      <td>10.744000000000002</td>
      <td>6.562</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1000.0</td>
      <td>12.354000000000001</td>
      <td>6.525</td>
    </tr>
  </tbody>
</table>
</div>



Now, our final dataset is ready we can perform the linear regression.


```
# Create feature matrix
X = df2.drop(['Weight'],axis = 'columns') 
# Create target vector
y = df2.Weight 

lm = linear_model.LinearRegression()
#TRain the model
lm.fit(X,y)
lm.score(X,y) 
# Note: We shouldnt use same dataset to check model score, this is out of scope of this tutorial.
```




    0.9058731241968216



This is how we can use pandas and sklearn library for performing the One Hot Encoding

 

