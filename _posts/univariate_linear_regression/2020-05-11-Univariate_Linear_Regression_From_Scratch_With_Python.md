---
title: 'Univariate Linear Regression From Scratch With Python'
date: 2020-05-11
permalink: /univariate_lr/
tags:
  - Univariate Linear Regression
  - Single Variable Linear Regression
  - Python
excerpt: This tutorial covers basic concepts of linear regression. I will explain the process of creating a model right from hypothesis function to gradient descent algorithm
---
![Linear_Regression_Header.png](https://raw.githubusercontent.com/satishgunjal/images/master/Linear_Regression_Header_640x441.png)

This tutorial covers basic concepts of linear regression. I will explain the process of creating a model right from hypothesis function to gradient descent algorithm. We will also use plots for better visualization of inner workings of the model. At the end we will test our model using single variable training data.


## **Introduction**
 
Linear regression is one of the most basic machine learning model. Its like a 'hello world' program of the machine learning. Linear regression is used when there is linear relationship between input variables and output variables. That means we can calculate the values of output variables by using some kind of linear combination of input variables. For example house prices are directly proportional to the house size, no of bedrooms, location etc. There is linear relationship between house prices and factors affecting it.
 
If there is only one input variable then we call it 'Single Variable Linear Regression' or **'Univariate Linear Regression'**. And in case of more than one input variables we call it 'Multi Variable Linear Regression' or **'Multivariate Linear Regression'**. In this tutorial we will work on univariate linear regression only. Linear regression is **'Supervised Learning Algorithm'** and mainly used to predict real valued output like house prices.


## **Objective Of Linear Model**
Every machine learning model actually generalize the relationship between input variables and output variables. In case of linear regression since relationship is linear, this generalization can be represented by simple line function. Let's consider the below example, input values are plotted on X axis and output values are plotted on Y axis.
 
![Linear Relationship Between Input(X) and Output(y) Values](https://raw.githubusercontent.com/satishgunjal/images/master/Only_Datapoints.png)
 
Since there are only few data point we can easily eyeball it and draw the best fit line, which will generalize the relationship between input and output variables for us. 
 
![Line Generalizing The Relationship Between Input(X) and Output(y) Values](https://raw.githubusercontent.com/satishgunjal/images/master/Datapoints_With_Line.png)
 
Since this line generalizes the relationship between input and output values for any prediction on given input value, we can simply plot it on a line and Y coordinate for that point will give us the prediction value. 
 
![Prediction Using Best Fit Line](https://raw.githubusercontent.com/satishgunjal/images/master/Datapoints_With_Line_And_Prediction.png)
 
So our objective is to find the best fit line which will generalize the given training data for future predictions.

## **Create Hypothesis Function**
Linear model hypothesis function is nothing but line function only.

Equation of line is 

**y = mx + b**

where
- m = slope/gradient
- x = input
- b = Y intercept

We are just going to use different notation to write it. I am using same notation and example data used in [Andrew Ng's Machine Learning course](https://www.coursera.org/learn/machine-learning/home/welcome)


**h(θ, x) = θ_0 + θ_1 * x_1**

where
- θ_1 = m = slope/gradient
- x = input
- θ_0 = b = Y intercept

![Line Function](https://raw.githubusercontent.com/satishgunjal/Images/master/Line_Function.png)

We already have input variables (x) with us if we can find θ_1 or m and θ_0 or b then we will get best fit line.

Since we have our hypothesis function we are now one step closer to our objective of finding best fit line. But how to find optimum values of theta parameters?

## **Create Cost Function**
It is obvious that to find the optimum values of theta parameters we have to try multiple values and then choose the best possible values based on the fit of the line. To do this we will create a cost function (J). Inner working of cost function is as below
- We will start with random values of θ_0 and θ_1
- We will execute the hypothesis function using theta values, to get the predicted values for every training example 
- Now we will compare our predicted values with actual target values from training data.
- The difference between prediction and actual values is called as 'cost'
- If our cost is close to 0 means our predictions are correct and so are our theta values
 
Instead of plain subtraction(predicted value - target value) we will use below form of more sophisticated cost function, also called as 'Square Error Function'
 
![Cost Function](https://raw.githubusercontent.com/satishgunjal/Images/master/Cost_Function_Formula.png)
 
![Square Error Calculation](https://raw.githubusercontent.com/satishgunjal/Images/master/Square_Error_Function.png)
 
 Cost function is a function of theta parameters. For simplicity if we plot the cost function against values of θ_1 then we get the 'Convex function'
 
![Convex Function](https://raw.githubusercontent.com/satishgunjal/Images/master/Convex_Function.png)
 
At the bottom of the curve we get the minimum value of cost for given value of θ_1. This process of trying different values theta to get minimum cost values is called as **'Minimizing The Cost'**.
 
Now we are one more step closure to our objective. Only the last part of puzzle is remaining. How many theta values should we try and how to change those values?

## **Gradient Descent Algorithm**
This is heart of our model, gradient descent algorithm will help us find optimum values of theta parameters. Inner working of gradient descent algorithm is as below,
  - We start with random value of θ_0 and θ_1
  - Calculate the cost using cost function
  - Change the value of θ_0 and θ_1 in order to find minimum cost value
  - Keep doing this till we get the minimum value of cost
 
In order to change the value of theta it's important to know whether to increase or decrease its value and by how much margin. Remember our cost function is a convex function and our objective is to go to its bottom. 
Partial derivative of the cost function will give us the slope at that point.
 
Consider below examples of positive and negatives slopes,
 
In case of positive slope we have to decrease the value of θ_1 to get minimum cost value.
 
![Positive Slope](https://raw.githubusercontent.com/satishgunjal/Images/master/Positive_Slope.png)
 
In case of negative slope we have to increase the value of θ_1 to get minimum cost value.
 
![Negative Slope](https://raw.githubusercontent.com/satishgunjal/Images/master/Negative_Slope.png)
 
So with the help of slope, we can decide whether to increase or decrease the theta value, but to control the magnitude of change we are going to use **'Learning Parameter Alpha (α)'**.
 
So the final formula to change the theta value is as below,
 
**θ_0 = θ_0 - alpha * partial derivative of cost function w.r.t θ_0**
 
**θ_1 = θ_1 - alpha * partial derivative of cost function w.r.t θ_1**
 
After replacing the value of partial derivative of cost function, our formula to get theta values will looks like,
 
![Gradient Descent Formula](https://raw.githubusercontent.com/satishgunjal/Images/master/Gradient_Descent_Formula.png)
 
Since at every step of gradient descent we are calculating the cost using all the training example, it is also called as **'Batch Gradient Descent'** algorithm
 
Enough of theory, now lets implement gradient descent algorithm using Python and create our linear model 

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
* We are going to use 'profits_and_populations_from_the_cities.csv' CSV file
* File contains two columns, the first column is the population of a city and the second column is the profit of a food truck in that city. A negative value for profit indicates a loss.



```
df = pd.read_csv('https://raw.githubusercontent.com/satishgunjal/datasets/master/univariate_profits_and_populations_from_the_cities.csv')
df.head() # To get first n rows from the dataset default value of n is 5
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
      <th>population</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.1101</td>
      <td>17.5920</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.5277</td>
      <td>9.1302</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.5186</td>
      <td>13.6620</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.0032</td>
      <td>11.8540</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.8598</td>
      <td>6.8233</td>
    </tr>
  </tbody>
</table>
</div>




```
X = df.values[:, 0]  # get input values from first column
y = df.values[:, 1]  # get output values from second column
m = len(y) # Number of training examples
print('X = ', X[: 5]) # Show only first 5 records
print('y = ', y[: 5])
print('m = ', m)
```

    X =  [6.1101 5.5277 8.5186 7.0032 5.8598]
    y =  [17.592   9.1302 13.662  11.854   6.8233]
    m =  97
    

## **Understand the data**
* Population of City in 10,000s and Profit in $10,000s. i.e 10K is multiplier for each data point
* There are total 97 training examples (m= 97 or 97 no of rows)
* There is only one feature (one column of feature and one of label/target/y)

## **Data Visualization**
* Lets assign the features(independent variables) values to variable X and target(dependent variable) values to variable y
* For this dataset, we can use a scatter plot to visualize the
data, since it has only two properties to plot (profit and population). 
* Many other problems that you will encounter in real life are multi-dimensional and can't be plotted on a 2D plot


```
plt.scatter(X,y, color='red',marker= '+')
plt.grid()
plt.rcParams["figure.figsize"] = (10,6)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Scatter plot of training data')
```

![png](https://raw.githubusercontent.com/satishgunjal/satishgunjal.github.io/master/_posts/univariate_linear_regression/images/Univariate_Linear_Regression_From_Scratch_With_Python_14_1.png)


## **Create Cost Function**
* Whenever possible we are going to use matrix calculation for better performance.
* Our hypothesis function is h(θ, x) = θ_0 + θ_1 * x_1 
 
 
### **Vector Representation Of Hypothesis Function**
 
* Please note we have to use above function for every value of training example to get predicted value. 
* Using for loop to do the calculations will take lots of time. Matrix operations are much faster than loops. 
* If we represent the input values(X) and theta values in matrix format then **Xθ** (X matrix multiply by θ vector) will give us predicted value for every training example.
* This is also called as vector implementation of equation.
 
* Lets create X and theta matrix using available values. Dimension of X matrix is (2 x 1)

  ![X_matrix_2x1.png](https://raw.githubusercontent.com/satishgunjal/Images/master/X_matrix_2x1.png)

* Lets create θ matrix using available values. Dimension of θ matrix is (2 x 1)

  ![theta_matrix_2x1.png](https://raw.githubusercontent.com/satishgunjal/Images/master/theta_matrix_2x1.png)
 
* If we want to do Xθ then no of columns of matrix X should match with no of rows of matrix θ. So let's add column of ones to matrix X to accommodate the θ_0 intercept term
 
  ![X_matrix_2x2.png](https://raw.githubusercontent.com/satishgunjal/Images/master/X_matrix_2x2.png)
 
* Since dimension of X matrix is now (2 x 2), we can perform the multiplication
  
  ![X_multiply_theta.png](https://raw.githubusercontent.com/satishgunjal/Images/master/X_multiply_theta.png)
 
* Product of Xθ will be a vector or 1D array,

  ![X_multiply_theta_result.PNG](https://raw.githubusercontent.com/satishgunjal/Images/master/X_multiply_theta_result.PNG)
 
* Notice that every row of our result vector is nothing but hypothesis function for every training example
* Note that as of now variables X and y are 1D arrays, we are going to convert them into matrices i.e. 2D array(list of a list). Python doesn't have a built-in type for matrices. However, we can treat 2D array as a matrix 



```
#Lets create a matrix with single column of ones
X_0 = np.ones((m, 1))
X_0[:5]
```




    array([[1.],
           [1.],
           [1.],
           [1.],
           [1.]])




```
# Using reshape function convert X 1D array to 2D array of dimension 97x1
X_1 = X.reshape(m, 1)
X_1[:5]
```




    array([[6.1101],
           [5.5277],
           [8.5186],
           [7.0032],
           [5.8598]])




```
# Lets use hstack() function from numpy to stack X_0 and X_1 horizontally (i.e. column wise) to make a single 2D array. 
# This will be our final X matrix (feature matrix)
X = np.hstack((X_0, X_1))
X[:5]
```




    array([[1.    , 6.1101],
           [1.    , 5.5277],
           [1.    , 8.5186],
           [1.    , 7.0032],
           [1.    , 5.8598]])



Remember to start with we need to initialize the theta parameter with random values. Lets initialize them with 0 values


```
theta = np.zeros(2)
theta
```




    array([0., 0.])



### **Now lets write a function to compute a cost**
* numpy.dot() this function returns the dot product of two arrays. For 2-D vectors, it is the equivalent to matrix multiplication
* numpy.subtract() this function perform the element wise subtraction
* numpy.square() this function perform the element wise square




```
def compute_cost(X, y, theta):
  """
  Compute cost for linear regression.

  Input Parameters
  ----------------
  X : 2D array where each row represent the training example and each column represent the feature ndarray. Dimension(m x n)
      m= number of training examples
      n= number of features (including X_0 column of ones)
  y : 1D array of labels/target value for each traing example. dimension(1 x m)

  theta : 1D array of fitting parameters or weights. Dimension (1 x n)

  Output Parameters
  -----------------
  J : Scalar value.
  """
  predictions = X.dot(theta)
  #print('predictions= ', predictions[:5])
  errors = np.subtract(predictions, y)
  #print('errors= ', errors[:5]) 
  sqrErrors = np.square(errors)
  #print('sqrErrors= ', sqrErrors[:5]) 
  J = 1 / (2 * m) * np.sum(sqrErrors)

  return J
```


```
# Lets compute the cost for theta values
cost = compute_cost(X, y, theta)
print('The cost for given values of theta_0 and theta_1 =', cost)
```

    The cost for given values of theta_0 and theta_1 = 32.072733877455676
    

The value of cost is 32.1, which is quite high. That means we have to keep trying different set of theta values till we get minimum cost value.

## **Gradient Descent Function**
* As of now our compute_cost() function is ready which returns cost for given values of theta
* We will create gradient_descent() function. In this function we are running a loop and for every iteration we are computing the value of theta using batch gradient descent algorithm. And using this value of theta we're also computing the 'cost' using cost function and storing it in a list.
* If our algorithm is working properly and given parameters(alpha and theta) are correct then value of 'cost' should decrease for every step.
* At the end we should get minimum cost value and corresponding theta parameters


```
def gradient_descent(X, y, theta, alpha, iterations):
  """
  Compute cost for linear regression.

  Input Parameters
  ----------------
  X : 2D array where each row represent the training example and each column represent the feature ndarray. Dimension(m x n)
      m= number of training examples
      n= number of features (including X_0 column of ones)
  y : 1D array of labels/target value for each traing example. dimension(m x 1)
  theta : 1D array of fitting parameters or weights. Dimension (1 x n)
  alpha : Learning rate. Scalar value
  iterations: No of iterations. Scalar value. 

  Output Parameters
  -----------------
  theta : Final Value. 1D array of fitting parameters or weights. Dimension (1 x n)
  cost_history: Conatins value of cost for each iteration. 1D array. Dimansion(m x 1)
  """
  cost_history = np.zeros(iterations)

  for i in range(iterations):
    predictions = X.dot(theta)
    #print('predictions= ', predictions[:5])
    errors = np.subtract(predictions, y)
    #print('errors= ', errors[:5])
    sum_delta = (alpha / m) * X.transpose().dot(errors);
    #print('sum_delta= ', sum_delta[:5])
    theta = theta - sum_delta;

    cost_history[i] = compute_cost(X, y, theta)  

  return theta, cost_history
```

Lets update the gradient descent learning parameters alpha and no of iterations


```
theta = [0., 0.]
iterations = 1500;
alpha = 0.01;
```


```
theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)
print('Final value of theta =', theta)
print('cost_history =', cost_history)
```

    Final value of theta = [-3.63029144  1.16636235]
    cost_history = [6.73719046 5.93159357 5.90115471 ... 4.48343473 4.48341145 4.48338826]
    

## **Visualization**
* Our algorithm returned the final values of theta, and as per our objective these values should give us best fit line
* Lets plot the line using predicted values.
* As per our vector implementation of hypothesis function we will get the predicted values using Xθ. Here θ is the value returned by our Gradient Descent Algorithm


```
# Since X is list of list (feature matrix) lets take values of column of index 1 only
plt.scatter(X[:,1], y, color='red', marker= '+', label= 'Training Data')
plt.plot(X[:,1],X.dot(theta), color='green', label='Linear Regression')

plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Linear Regression Fit')
plt.legend()
```

![png](https://raw.githubusercontent.com/satishgunjal/satishgunjal.github.io/master/_posts/univariate_linear_regression/images/Univariate_Linear_Regression_From_Scratch_With_Python_31_1.png)


## **Convergence of Gradient Descent**
* cost_history contains the values of cost for every iteration performed during batch gradient descent
* If all our parameters are correct then cost should reduce for every iteration(step)
* lets plot the values of cost against no of iterations to visualize the performance of the Gradient Descent Algorithm
 




```
plt.plot(range(1, iterations + 1),cost_history, color='blue')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')
```

![png](https://raw.githubusercontent.com/satishgunjal/satishgunjal.github.io/master/_posts/univariate_linear_regression/images/Univariate_Linear_Regression_From_Scratch_With_Python_33_1.png)


## **Testing the model**
* Predict the profit for population 35,000
* Predict the profit for population 70,000
 
### **Manual Calculations**
* Generally 'theta' are referred as weights or coefficients.
* Values of theta/weights/coefficients are [-3.63029144  1.16636235]
* Hypothesis function is h(θ, x) = θ_0 * x_0 + θ_1 * x_1
* Given values are
  - θ_0 = -3.63029144,
  - θ_1 = 1.16636235,
  - x_0 = 1,
  - x_1= 3.5 (remember all our values are in multiples ok 10,000)
 
* h(x) = (-3.63029144 * 1) + (1.16636235 * 3.5)
* h(x) = 0.4519767849999998
* Since all our values are in multiples of 10,000
* h(x) = 0.4519767849999998 * 10000
* h(x) = 4519.767849999998
* For population = 35,000, we predict a profit of 4519.7678677
 
We can predict the result using our model as below
 
 




```
predict1 = np.array([1, 3.5]).dot(theta)
print("For population = 35,000, our prediction of profit is", predict1 * 10000)

predict2 = np.array([1, 7]).dot(theta)
print("For population = 70,000, our prediction of profit is", predict2 * 10000)
```

    For population = 35,000, our prediction of profit is 4519.7678677017675
    For population = 70,000, our prediction of profit is 45342.45012944714
    
## **Conclusion**

This concludes our univariate linear regression. But in real life profit of food truck also depends on lots of many other factors. We can use the same algorithm implemented above to perform linear regression when there are multiple factors affecting output value. In next tutorial I will explain the multivariate linear regression.
