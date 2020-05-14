![multivariate_linear_regression.png](https://github.com/satishgunjal/images/blob/master/multivariate_linear_Regression_header.png?raw=true)
 
In this tutorial we are going to cover linear regression with multiple input variables. We are going to use same model that we have created in [Univariate Linear Regression](https://satishgunjal.github.io/univariate_lr/) tutorial. I would recommend to read Univariate Linear Regression tutorial first.
We will define the hypothesis function with multiple variables and use gradient descent algorithm. We will also use plots for better visualization of inner workings of the model. At the end we will test our model using training data.
 
## **Introduction**
 
In case of multivariate linear regression output value is dependent on multiple input values. The relationship between input values, format of different input values and range of input values plays important role in linear model creation and prediction. I am using same notation and example data used in [Andrew Ng's Machine Learning course](https://www.coursera.org/learn/machine-learning/home/welcome)
 
 ## **Hypothesis Function**
 
Our hypothesis function for univariate linear regression was 
 
 
```
h(x) = theta_0 + theta_1*x_1 
where x_1 is only input value
```
For multiple input value, hypothesis function will look like,
 
```
h(x) = theta_0 + theta_1 * x_1 + theta_2 * x_2 .....theat_n * x_n
where x_1, x_2...x_n are multiple input valus
```
If we consider the house price example then the factors affecting its price like house size, no of bedrooms, location etc are nothing but input variables of above hypothesis function.
 
## **Cost Function**
 
Our cost function remains same as used in Univariate linear regression
 
![Cost Function](https://raw.githubusercontent.com/satishgunjal/Images/master/Cost_Function_Formula.png)
 
 
For more details about cost function please refer 'Create Cost Function' section of [Univariate Linear Regression](https://satishgunjal.github.io/univariate_lr/)
 
## **Gradient Descent Algorithm**
 
Gradient descent algorithm function format remains same as used in Univariate linear regression. But here we have to do it for all the theta values(no of theta values = no of features + 1).
 
![Gradient Descent Formula n Features](https://raw.githubusercontent.com/satishgunjal/Images/master/gradient_descent_formula_n_features.PNG)
 
For more details about gradient descent algorithm please refer 'Gradient Descent Algorithm' section of [Univariate Linear Regression](https://satishgunjal.github.io/univariate_lr/)

# **Python Code**

## **Notations used**
* m   = no of training examples (no of rows of feature matrix)
* n   = no of features (no of columns of feature matrix)
* x's = input variables / independent variables / features
* y's = output variables / dependent variables / target

## **Import the required libraries**
* numpy : Numpy is the core library for scientific computing in Python. It is used for working with arrays and matrices.
* pandas: Used for data manipulation and analysis
* matplotlib : It's plotting library, and we are going to use it for data visualization

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## **Load the data**
* We are going to use 'multivariate_housing_prices_in_portlans_oregon.csv' CSV file
* File contains three columns 'size(in square feet)',	'number of bedrooms' and 'price'
