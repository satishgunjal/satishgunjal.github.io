---
title: 'Multivariate Linear Regression Using Scikit Learn'
date: 2020-05-24
permalink: /multivariate_lr_scikit/
tags:
  - Multivariate Linear Regression
  - Multiple Variable Linear Regression
  - Python
excerpt: In this tutorial we are going to use the Linear Models from Sklearn library. Scikit-learn is one of the most popular open source machine learning library for python.
---

![multivariate_linear_Regression_scikit_header.png](https://raw.githubusercontent.com/satishgunjal/images/master/multivariate_linear_Regression_scikit_header.png)
 
In this tutorial we are going to use the Linear Models from Sklearn library. We are also going to use the same test data used in [Multivariate Linear Regression From Scratch With Python](http://satishgunjal.github.io/multivariate_lr/) tutorial

## **Introduction**
 
Scikit-learn is one of the most popular open source machine learning library for python. It provides range of machine learning models, here we are going to use linear model. Sklearn linear models are used when target value is some kind of linear combination of input value. Sklearn library has multiple types of linear models to choose form. The way we have implemented the 'Batch Gradient Descent' algorithm in [Multivariate Linear Regression From Scratch With Python](http://satishgunjal.github.io/multivariate_lr/) tutorial, every Sklearn linear model also use specific mathematical model to find the best fit line. 
 
## **Hypothesis Function Comparison**
 
The hypothesis function used by Linear Models of Sklearn library is as below
 
y(w, x) = w_0 + (w_1 * x_1) + (w_2 * x_2) .......(w_n * x_n)
 
Where,
* y(w, x) = Target/output value
* x_1 to x_n = Dependent/Input value
* w_0 = intercept 
* w_1 to w_n =  as coef for every input feature(x_1 to x_n)
 
You must have noticed that above hypothesis function is not matching with the hypothesis function used in [Multivariate Linear Regression From Scratch With Python](http://satishgunjal.github.io/multivariate_lr/) tutorial. Actually both are same, just different notations are used

h(θ, x) = θ_0 + (θ_1 * x_1) + (θ_2 * x_2)......(θ_n * x_n)
 
Where,
 
* Both the hypothesis function use 'x' to represent input values or features
* y(w, x) =  h(θ, x) = Target or output value
* w_0 = θ__0  =  intercept or Y intercept
* w_1 to w_n = θ__1 to θ__n =  coef or slope/gradient

## **Python Code**
 
Yes, we are jumping to coding right after hypothesis function, because we are going to use Sklearn library which has multiple algorithms to choose from.
 
## **Import the required libraries**
* numpy : Numpy is the core library for scientific computing in Python. It is used for working with arrays and matrices.
* pandas: Used for data manipulation and analysis
* matplotlib : It’s plotting library, and we are going to use it for data visualization
* linear_model: Sklearn linear regression model 
 
*In case you don't have any experience using these libraries, don't worry I will explain every bit of code for better understanding*




```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
```

## **Load the data**
* We are going to use 'multivariate_housing_prices_in_portlans_oregon.csv' CSV file
* File contains three columns 'size(in square feet)',	'number of bedrooms' and 'price'


```
df = pd.read_csv('https://raw.githubusercontent.com/satishgunjal/datasets/master/multivariate_housing_prices_in_portlans_oregon.csv')
print('Dimension of dataset= ', df.shape)
df.head() # To get first n rows from the dataset default value of n is 5
```

    Dimension of dataset=  (47, 3)
    




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
      <th>size(in square feet)</th>
      <th>number of bedrooms</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2104</td>
      <td>3</td>
      <td>399900</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1600</td>
      <td>3</td>
      <td>329900</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2400</td>
      <td>3</td>
      <td>369000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1416</td>
      <td>2</td>
      <td>232000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3000</td>
      <td>4</td>
      <td>539900</td>
    </tr>
  </tbody>
</table>
</div>




```
X = df.values[:, 0:2]  # get input values from first two columns
y = df.values[:, 2]  # get output values from last coulmn
m = len(y) # Number of training examples

print('Total no of training examples (m) = %s \n' %(m))

# Show only first 5 records
for i in range(5):
    print('X =', X[i, ], ', y =', y[i])
```

    Total no of training examples (m) = 47 
    
    X = [2104    3] , y = 399900
    X = [1600    3] , y = 329900
    X = [2400    3] , y = 369000
    X = [1416    2] , y = 232000
    X = [3000    4] , y = 539900
    

## **Understand the data**
 
* There are total 47 training examples (m= 47 or 47 no of rows)
* There are two features (two columns of feature and one of label/target/y)
* Total no of features (n) = 2

 
### **Feature Normalization**
 
* As you can notice size of the house and no of bedrooms are not in same range(house sizes are about 1000 times the number of bedrooms).
* Sklearn provides libraries to perform the feature normalization. We don't have to write our own function for that.
* During model training we will enable the feature normalization
* To know more about feature normalization please refer 'Feature Normalization' section in [Multivariate Linear Regression From Scratch With Python](http://satishgunjal.github.io/multivariate_lr/) tutorial
 



## **Which Sklearn Linear Regression Algorithm To Choose**
 
* Sklearn library have multiple linear regression algorithms
* Note: The way we have implemented the cost function and gradient descent algorithm in previous tutorials every Sklearn algorithm also have some kind of mathematical model.
* Different algorithms are better suited for different types of data and type of problems
* Flow chart below will give you brief idea on how to choose right algorithm
 
  ![Choosing_Right_Sklearn_Linear_Model.png](https://raw.githubusercontent.com/satishgunjal/images/master/Choosing%20Right%20Sklearn%20Linear%20Model.png)
 



## **Ordinary Least Squares Algorithm**

* This is one of the most basic linear regression algorithm. 
* Mathematical formula used by ordinary least square algorithm is as below,

  ![ordinary_least_squares_formlua.png](https://github.com/satishgunjal/images/blob/master/ordinary_least_squares_formlua_1.png?raw=true)

* The objective of Ordinary Least Square Algorithm is to minimize the residual sum of squares. Here the term residual means 'deviation of predicted value(Xw) from actual value(y)'
* Problem with ordinary least square model is size of coefficients increase exponentially with increase in model complexity



```
model_ols =  linear_model.LinearRegression(normalize=True)
model_ols.fit(X,y) 
# fit() method is used for training the model
# Note the first parameter(feature) is must be 2D array(feature matrix)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=True)



### **Understanding Training Results**
* Note: If training is successful then we get the result like above. Where all the default values used by LinearRgression() model are displayed.
* Note that 'normalization = true'
* As per our hypothesis function, 'model' object contains the coef and intercept values


```
coef = model_ols.coef_
intercept = model_ols.intercept_
print('coef= ', coef)
print('intercept= ', intercept)
```

    coef=  [  139.21067402 -8738.01911233]
    intercept=  89597.90954279757
    

Note that for every feature we get the coefficient value. Since we have two features(size and no of bedrooms) we get two coefficients. Magnitude and direction(+/-) of all these values affect the prediction results.

### **Visualization**
* Check below table for comparison between price from dataset and predicted price by our model
* We will also plot the scatter plot of price from dataset vs predicted weight
 
**Note: Here we are using the same dataset for training the model and to do predictions. Recommended way is to split the dataset and use 80% for training and 20% for testing the model. We will learn more about this in future tutorials.**



```
predictedPrice = pd.DataFrame(model_ols.predict(X), columns=['Predicted Price']) # Create new dataframe of column'Predicted Price'
actualPrice = pd.DataFrame(y, columns=['Actual Price'])
actualPrice = actualPrice.reset_index(drop=True) # Drop the index so that we can concat it, to create new dataframe
df_actual_vs_predicted = pd.concat([actualPrice,predictedPrice],axis =1)
df_actual_vs_predicted.T
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
      <th>40</th>
      <th>41</th>
      <th>42</th>
      <th>43</th>
      <th>44</th>
      <th>45</th>
      <th>46</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual Price</th>
      <td>399900.000000</td>
      <td>329900.000000</td>
      <td>369000.000000</td>
      <td>232000.000000</td>
      <td>539900.000000</td>
      <td>299900.000000</td>
      <td>314900.000000</td>
      <td>198999.000000</td>
      <td>212000.00000</td>
      <td>242500.000000</td>
      <td>239999.000000</td>
      <td>347000.000000</td>
      <td>329999.000000</td>
      <td>699900.000000</td>
      <td>259900.00000</td>
      <td>449900.000000</td>
      <td>299900.000000</td>
      <td>199900.000000</td>
      <td>499998.000000</td>
      <td>599000.000000</td>
      <td>252900.000000</td>
      <td>255000.000000</td>
      <td>242900.00000</td>
      <td>259900.000000</td>
      <td>573900.000000</td>
      <td>249900.000000</td>
      <td>464500.000000</td>
      <td>469000.000000</td>
      <td>475000.000000</td>
      <td>299900.00000</td>
      <td>349900.000000</td>
      <td>169900.000000</td>
      <td>314900.000000</td>
      <td>579900.000000</td>
      <td>285900.000000</td>
      <td>249900.000000</td>
      <td>229900.000000</td>
      <td>345000.000000</td>
      <td>549000.000000</td>
      <td>287000.00000</td>
      <td>368500.000000</td>
      <td>329900.000000</td>
      <td>314000.000000</td>
      <td>299000.000000</td>
      <td>179900.000000</td>
      <td>299900.000000</td>
      <td>239500.000000</td>
    </tr>
    <tr>
      <th>Predicted Price</th>
      <td>356283.110339</td>
      <td>286120.930634</td>
      <td>397489.469848</td>
      <td>269244.185727</td>
      <td>472277.855146</td>
      <td>330979.021018</td>
      <td>276933.026149</td>
      <td>262037.484029</td>
      <td>255494.58235</td>
      <td>271364.599188</td>
      <td>324714.540688</td>
      <td>341805.200241</td>
      <td>326492.026099</td>
      <td>669293.212232</td>
      <td>239902.98686</td>
      <td>374830.383334</td>
      <td>255879.961021</td>
      <td>235448.245292</td>
      <td>417846.481605</td>
      <td>476593.386041</td>
      <td>309369.113195</td>
      <td>334951.623863</td>
      <td>286677.77333</td>
      <td>327777.175516</td>
      <td>604913.374134</td>
      <td>216515.593625</td>
      <td>266353.014924</td>
      <td>415030.014774</td>
      <td>369647.335045</td>
      <td>430482.39959</td>
      <td>328130.300837</td>
      <td>220070.564448</td>
      <td>338635.608089</td>
      <td>500087.736599</td>
      <td>306756.363739</td>
      <td>263429.590769</td>
      <td>235865.877314</td>
      <td>351442.990099</td>
      <td>641418.824078</td>
      <td>355619.31032</td>
      <td>303768.432883</td>
      <td>374937.340657</td>
      <td>411999.633297</td>
      <td>230436.661027</td>
      <td>190729.365581</td>
      <td>312464.001374</td>
      <td>230854.293049</td>
    </tr>
  </tbody>
</table>
</div>




```
plt.scatter(y, model_ols.predict(X))
plt.xlabel('Price From Dataset')
plt.ylabel('Price Predicted By Model')
plt.rcParams["figure.figsize"] = (10,6) # Custom figure size in inches
plt.title("Price From Dataset Vs Price Predicted By Model")
```


![png](https://raw.githubusercontent.com/satishgunjal/images/master/price_from_dataset_vs_price_predicted_by_model.png)


## **Testing the model**
* **Question: Estimate the price of a 1650 sq-ft, 3-bedroom house**
* We can simply use 'predict()' of sklearn library to predict the price of the house


```
price = model_ols.predict([[2104,    3]])
print('Predicted price of a 1650 sq-ft, 3 br house:', price)
```

    Predicted price of a 1650 sq-ft, 3 br house: [356283.1103389]
    

## **Ridge Regression Algorithm**
* Ridge regression addresses some problems of Ordinary Least Squares by imposing a penalty on the size of the coefficients
* Ridge model uses complexity parameter alpha to control the size of coefficients
* Note: alpha should be more than '0', or else it will perform same as ordinary linear square model
* Mathematical formula used by Ridge Regression algorithm is as below,
 
  ![ridge_regression_formlua.png](https://github.com/satishgunjal/images/blob/master/ridge_regression_formlua_1.png?raw=true)
 




```
model_r = linear_model.Ridge(normalize= True, alpha= 35)
model_r.fit(X,y)
print('coef= ' , model_r.coef_)
print('intercept= ' , model_r.intercept_)
price = model_r.predict([[2104,    3]])
print('Predicted price of a 1650 sq-ft, 3 br house:', price)
```

    coef=  [   3.70764427 1958.37472904]
    intercept=  326786.38211867993
    Predicted price of a 1650 sq-ft, 3 br house: [340462.38984537]
    

## **LASSO Regression Algorithm**
* Similar to Ridge regression LASSO also uses regularization parameter alpha but it estimates sparse coefficients i.e. more number of 0 coefficients
* That's why its best suited when dataset contains few important features
* LASSO model uses regularization parameter alpha to control the size of coefficients
* Note: alpha should be more than '0', or else it will perform same as ordinary linear square model
* Mathematical formula used by LASSO Regression algorithm is as below,
 
  ![lasso_regression_formlua.png](https://github.com/satishgunjal/images/blob/master/lasso_regression_formlua_1.png?raw=true)




```
model_l = linear_model.Lasso(normalize= True, alpha= 0.55)
model_l.fit(X,y)
print('coef= ' , model_l.coef_)
print('intercept= ' , model_l.intercept_)
price = model_l.predict([[2104,    3]])
print('Predicted price of a 1650 sq-ft, 3 br house:', price)
```

    coef=  [  139.19963776 -8726.55682971]
    intercept=  89583.65169819258
    Predicted price of a 1650 sq-ft, 3 br house: [356280.01905528]
    

## **Conclusion**
 
As you can notice with Sklearn library we have very less work to do and everything is handled by library. We don't have to add column of ones, no need to write our cost function or gradient descent algorithm. We can directly use library and tune the hyper parameters (like changing the value of alpha) till the time we get satisfactory results. If you are following my machine learning tutorials from the beginning then implementing our own gradient descent algorithm and then using prebuilt models like Ridge or LASSO gives us very good perspective of inner workings of these libraries and hopeful it will help you understand it better.
