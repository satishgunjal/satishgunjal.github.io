---
title: 'Time Series Analysis and Forecasting (ARIMA)'
date: 2020-07-03
permalink: /time_series/
tags:
  - Time Series
  - Statsmodels
  - ARIMA
  - Machine Learning
  - Data Science
  - Predictive Analytics
header:
  teaser: https://raw.githubusercontent.com/satishgunjal/images/master/Time_Series_Header_1000x690.png
excerpt: Any data recorded with some fixed interval of time is called as time series data. This fixed interval can be hourly, daily, monthly or yearly. Objective of time series analysis is to understand how change in time affect the dependent variables and accordingly predict values for future time intervals.

---

![Time_Series_Header](https://raw.githubusercontent.com/satishgunjal/images/master/Time_Series_Header_1000x690.png)

# Table of Contents

* [What is Time Series](#1)
  - [Time Series Characteristics](#2)
    + [Trend](#3) 
    + [Seasonality](#4)
    + [Irregularities](#5)
    + [Cyclicity](#6)
* [Time Series Analysis](#7)
  - [Decomposition of Time Series](#8)
  - [Stationary Data](#9)
  - [Test for Stationarity](#10)
    + [Rolling Statistics](#11)
    + [Augmented Dickey Fuller Test](#12)
  - [Convert Non Stationary Data to Stationary Data](#13)
    + [Differencing](#14)
    + [Transformation](#15)
    + [Moving Average](#16)
      + [Weighted Moving Averages(WMA)](#17)
      + [Centered Moving Averages(CMS)](#18)
      + [Trailing Moving Averages(TMA)](#19)
  - [Correlation](#20)
    + [ACF: Auto Correlation Function](#21)
    + [PACF: Partial Auto Correlation Function](#22)
* [Time Series Forecasting](#23)
  - [Models Used For Time Series Forecasting](#24)
  - [ARIMA](#25)
    + [Auto-Regressive (AR) Model](#26)
    + [Integration(I)](#27)
    + [Moving Average (MA) Model](#28)
* [Python Example](#29)

# What is Time Series <a id ="1"></a>
Any data recorded with some fixed interval of time is called as time series data. This fixed interval can be hourly, daily, monthly or yearly. e.g. hourly temp reading, daily changing fuel prices, monthly electricity bill, annul company profit report etc. In time series data, time will always be independent variable and there can be one or many dependent variable. 

Sales forecasting time series with shampoo sales for every month will look like this, 

![Shampoo_Sales](https://raw.githubusercontent.com/satishgunjal/images/master/Shampoo_Sales.png)

In above example since there is only one variable dependent on time so its called as univariate time series. If there are multiple dependent variables, then its called as multivariate time series.

Objective of time series analysis is to understand how change in time affect the dependent variables and accordingly predict values for future time intervals.


## Time Series Characteristics <a id ="2"></a>
Mean, standard deviation and seasonality defines different characteristics of the time series. 

![Time_Series_Characteristics](https://raw.githubusercontent.com/satishgunjal/images/master/Time_Series_Characteristics.png)

Important characteristics of the time series are as below

### Trend <a id ="3"></a>
Trend represent the change in dependent variables with respect to time from start to end. In case of increasing trend dependent variable will increase with time and vice versa. It's not necessary to have definite trend in time series, we can have a single time series with increasing and decreasing trend. In short trend represent the varying mean of time series data.

![Trend](https://raw.githubusercontent.com/satishgunjal/images/master/Trend.png)

### Seasonality <a id ="4"></a>
If observations repeats after fixed time interval then they are referred as seasonal observations. These seasonal changes in data can occur because of natural events or man-made events. For example every year warm cloths sales increases just before winter season. So seasonality represent the data variations at fixed intervals.

![Seasonality](https://raw.githubusercontent.com/satishgunjal/images/master/Seasonality.png)

### Irregularities <a id ="5"></a>
This is also called as noise. Strange dips and jump in the data are called as irregularities. These fluctuations are caused by uncontrollable events like earthquakes, wars, flood, pandemic etc. For example because of COVID-19 pandemic there is huge demand for hand sanitizers and masks.

![Irregularities](https://raw.githubusercontent.com/satishgunjal/images/master/Irregularities.png)

### Cyclicity <a id ="6"></a>
Cyclicity occurs when observations in the series repeats in random pattern. Note that if there is any fixed pattern then it becomes seasonality, in case of cyclicity observations may repeat after a week, months or may be after a year. These kinds of patterns are much harder to predict.

![Cyclicity](https://raw.githubusercontent.com/satishgunjal/images/master/Cyclicity.png)

Time series data which has above characteristics is called as 'Non-Stationary Data'. For any analysis on time series data we must convert it to 'Stationary Data'

The general guideline is to estimate the trend and seasonality in the time series, and then make the time series stationary for data modeling. In data modeling step statistical techniques are used for time series analysis and forecasting. Once we have the predictions, in the final step forecasted values converted into the original scale by applying trend and seasonality constraints back.


# Time Series Analysis <a id ="7"></a>
As name suggest its analysis of the time series data to identify the patterns in it. I will briefly explain the different techniques and test for time series data analysis.

## Decomposition of Time Series <a id ="8"></a>
Time series decomposition helps to deconstruct the time series into several component like trend and seasonality for better visualization of its characteristics. Using time-series decomposition makes it easier to quickly identify a changing mean or variation in the data

![Decomposition_of_Time_Series](https://raw.githubusercontent.com/satishgunjal/images/master/Decomposition_of_Time_Series.png)

## Stationary Data <a id ="9"></a>
For accurate analysis and forecasting trend and seasonality is removed from the time series and converted it into stationary series.
Time series data is said to be stationary when statistical properties like mean, standard deviation are constant and there is no seasonality. In other words statistical properties of the time series data should not be a function of time.

![Stationarity](https://raw.githubusercontent.com/satishgunjal/images/master/Stationarity.png)

## Test for Stationarity <a id ="10"></a>
Easy way is to look at the plot and look for any obvious trend or seasonality. While working on real world data we can also use more sophisticated methods like rolling statistic and Augmented Dickey Fuller test to check stationarity of the data. 

### Rolling Statistics <a id ="11"></a>
In rolling statistics technique we define a size of window to calculate the mean and standard deviation throughout the series. For stationary series mean and standard deviation shouldn't change with time.

### Augmented Dickey Fuller (ADF) Test <a id ="12"></a>
I won't go into the details of how this test works. I will concentrate more on how to interpret the result of this test to determine the stationarity of the series. ADF test will return 'p-value' and 'Test Statistics' output values.
* **p-value > 0.05**: non-stationary.
* **p-value <= 0.05**: stationary.
* **Test statistics**: More negative this value more likely we have stationary series. Also, this value should be smaller than critical values(1%, 5%, 10%). For e.g. If test statistic is smaller than the 5% critical values, then we can say with 95% confidence that this is a stationary series

## Convert Non-Stationary Data to Stationary Data <a id ="13"></a>
Accounting for the time series data characteristics like trend and seasonality is called as making data stationary. So by making the mean and variance of the time series constant, we will get the stationary data. Below are the few technique used for the same…

### Differencing <a id ="14"></a>
Differencing technique helps to remove the trend and seasonality from time series data. Differencing is performed by subtracting the previous observation from the current observation. The differenced data will contain one less data point than original data. So differencing actually reduces the number of observations and stabilize the mean of a time series.

```
difference = previous observation - current observation
```
After performing the differencing it's recommended to plot the data and  visualize the change. In case there is not sufficient improvement you can perform second order or even third order differencing.

### Transformation <a id ="15"></a>
A simple but often effective way to stabilize the variance across time is to apply a power transformation to the time series. Log, square root, cube root are most commonly used transformation techniques.
Most of the time you can pick the type of growth of the time series and accordingly choose the transformation method. For. e.g. A time series that has a quadratic growth trend can be made linear by taking the square root. In case differencing don't work, you may first want to use one of above transformation technique to remove the variation from the series. 

![Log_Transformation](https://raw.githubusercontent.com/satishgunjal/images/master/Log_Transformation.png)

### Moving Average <a id ="16"></a>
In moving averages technique, a new series is created by taking the averages of data points from original series. In this technique we can use two or more raw data points to calculate the average. This is also called as 'window width (w)'. Once window width is decided, averages are calculated from start to the end for each set of w consecutive values, hence the name moving averages. It can also be used for time series forecasting.

![Moving_Average](https://raw.githubusercontent.com/satishgunjal/images/master/Moving_Average.png)

#### Weighted Moving Averages(WMA) <a id ="17"></a>
WMA is a technical indicator that assigns a greater weighting to the most recent data points, and less weighting to data points in the distant past. The WMA is obtained by multiplying each number in the data set by a predetermined weight and summing up the resulting values. There can be many techniques for assigning weights. A popular one is exponentially weighted moving average where weights are assigned to all the previous values with a decay factor.

#### Centered Moving Averages(CMS) <a id ="18"></a>
In a centered moving average, the value of the moving average at time t is computed by centering the window around time t and averaging across the w values within the window. For example, a center moving average with a window of 3 would be calculated as
  ```
  CMA(t) = mean(t-1, t, t+1)
  ```
  
CMA is very useful for visualizing the time series data
  
#### Trailing Moving Averages(TMA) <a id ="19"></a>
In trailing moving average, instead of averaging over a window that is centered around a time period of interest, it simply takes the average of the last w values. For example, a trailing moving average with a window of 3 would be calculated as:
 ```
 TMA(t) = mean(t-2, t-1, t)
 ```
 
 TMA are useful for forecasting.

## Correlation <a id ="20"></a>
* Most important point about values in time series is its dependence on the previous values.
* We can calculate the correlation for time series observations with previous time steps, called as lags.
* Because the correlation of the time series observations is calculated with values of the same series at previous times, this is called an autocorrelation or serial correlation.
* To understand it better lets consider the example of fish prices. We will use below notation to represent the fish prices. 
    - P(t)= Fish price of today
    - P(t-1) = Fish price of last month
    - P(t-2) =Fish price of last to last month
* Time series of fish prices can be represented as P(t-n),..... P(t-3), P(t-2),P(t-1), P(t)
* So if we have fish prices for last few months then it will be easy for us to predict the fish price for today (Here we are ignoring all other external factors that may affect the fish prices

All the past and future data points are related in time series and ACF and PACF functions help us to determine correlation in it.

### Auto Correlation Function (ACF) <a id ="21"></a>
* ACF tells you how correlated points are with each other, based on how many time steps they are separated by.
* Now to understand it better lets consider above example of fish prices. Let's try to find the correlation between fish price for current month P(t) and two months ago P(t-2). Important thing to note that, fish price of two months ago can directly affect the today's fish price or it can indirectly affect the fish price through last months price P(t-1)
* So ACF consider the direct as well indirect effect between the points while determining the correlation

### Partial Auto Correlation Function (PACF) <a id ="22"></a>
* Unlike ACF, PACF only consider the direct effect between the points while determining the correlation
* In case of above fish price example PACF will determine the correlation between fish price for current month P(t) and two months ago P(t-2) by considering only P(t) and P(t-2) and ignoring P(t-1)


# Time Series Forecasting <a id ="23"></a>
Forecasting refers to the future predictions based on the time series data analysis. Below are the steps performed during time series forecasting

* Step 1: Understand the time series characteristics like trend, seasonality etc
* Step 2: Do the analysis and identify the best method to make the time series stationary
* Step 3: Note down the transformation steps performed to make the time series stationary and make sure that the reverse transformation of data is possible to get the original scale back
* Step 4: Based on data analysis choose the appropriate model for time series forecasting
* Step 5: We can assess the performance of a model by applying simple metrics such as residual sum of squares(RSS). Make sure to use whole data for prediction.
* Step 6: Now we will have an array of predictions which are in transformed scale. We just need to apply the reverse transformation to get the prediction values in original scale.
* Step 7: At the end we can do the future forecasting and get the future forecasted values in original scale.

## Models Used For Time Series Forecasting <a id ="24"></a>
* Autoregression (AR)
* Moving Average (MA)
* Autoregressive Moving Average (ARMA)
* Autoregressive Integrated Moving Average (ARIMA)
* Seasonal Autoregressive Integrated Moving-Average (SARIMA)
* Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX)
* Vector Autoregression (VAR)
* Vector Autoregression Moving-Average (VARMA)
* Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX)
* Simple Exponential Smoothing (SES)
* Holt Winter’s Exponential Smoothing (HWES)

Next part of this article we are going to analyze and forecast air passengers time series data using ARIMA model. Brief introduction of ARIMA model is as below

## ARIMA <a id ="25"></a>
* ARIMA stands for Auto-Regressive Integrated Moving Averages. It is actually a combination of AR and MA model. 
* ARIMA has three parameters 'p' for the order of Auto-Regressive (AR) part, 'q' for the order of Moving Average (MA) part and 'd' for the order of integrated part. 

### Auto-Regressive (AR) Model: <a id ="26"></a>
* As the name indicates, its the regression of the variables against itself. In this model linear combination of the past values are used to forecast the future values. 
* To figure out the order of AR model we will use PACF function

### Integration(I): <a id ="27"></a>
* Uses differencing of observations (subtracting an observation from observation at the previous time step) in order to make the time series stationary. Differencing involves the subtraction of the current values of a series with its previous values d number of times.
* Most of the time value of d = 1, means first order of difference.

### Moving Average (MA) Model: <a id ="28"></a>
* Rather than using past values of the forecast variable in a regression, a moving average model uses linear combination of past forecast errors
* To figure out the order of MA model we will use ACF function

# Python Example <a id ="29"></a>
We have a monthly time series data of the air passengers from 1 Jan 1949 to 1 Dec 1960. Each row contains the air passenger number for a month of that particular year. Objective is to build a model to forecast the air passenger traffic for future months.

For source code please refer my  [kaggle kernel](https://www.kaggle.com/satishgunjal/tutorial-time-series-analysis-and-forecasting#Python-Example-)
