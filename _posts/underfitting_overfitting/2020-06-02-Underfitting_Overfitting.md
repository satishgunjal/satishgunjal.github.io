---
title: 'Underfitting & Overfitting'
date: 2020-06-02
permalink: /underfitting_overfitting/
tags:
  - Bias
  - Variance
header:
  teaser: https://raw.githubusercontent.com/satishgunjal/images/master/underfitting_overfitting_header_1.png
excerpt: Main objective of any machine learning model is to generalize the learning based on training data, so that it will be able to do predictions accurately on unknown data. Overfitting and underfitting models don't generalize well and results in poor performance.

---

![Underfitting_overfitting_header_1.png](https://raw.githubusercontent.com/satishgunjal/images/master/underfitting_overfitting_header_1.png)

Remember that the main objective of any machine learning model is to generalize the learning based on training data, so that it will be able to do predictions accurately on unknown data. As you can notice the words 'Overfitting' and 'Underfitting' are kind of opposite of the term 'Generalization'. Overfitting and underfitting models don't generalize well and results in poor performance.

## Underfitting
* Underfitting occurs when machine learning model don't fit the training data well enough. It is usually caused by simple function that cannot capture the underlying trend in the data.
* Underfitting models have high error in training as well as test set. This behavior is called as 'Low Bias'
* This usually happens when we try to fit linear function for non-linear data.
* Since underfitting models don't perform well on training set, it's very easy to detect underfitting

  ![Underfitting.png](https://raw.githubusercontent.com/satishgunjal/images/master/Underfitting.png)


### How To Avoid Underfitting?
* Increasing the model complexity. e.g. If linear function under fit then try using polynomial features
* Increase the number of features by performing the feature engineering

### Example
Please refer my [Multiple Linear Regression Fish Weight Prediction](https://www.kaggle.com/satishgunjal/multiple-linear-regression-fish-weight-prediction) Kaggle notebook. In this study I am using linear function, which is not fitting the data well. Though model score is on higher side, but one major issue with prediction is negative weight values. This behavior is true for smaller(less than 20gm) weight values.

## Overfitting
* Overfitting occurs when machine learning model tries to fit the training data too well. It is usually caused by complicated function that creates lots of unnecessary curves and angles that are not related with data and end up capturing the noise in data.
* Overfitting models have low error in training set but high error in test set. This behavior is called as 'High Variance'

  ![Overfitting.png](https://raw.githubusercontent.com/satishgunjal/images/master/Overfitting.png)

### How To Avoid Overfitting?
* Since overfitting algorithm captures the noise in data, reducing the number of features will help. We can manually select only important features or can use model selection algorithm for same
* We can also use the 'Regularization' technique. It works well when we have lots of slightly useful features. Sklearn linear model(Ridge and LASSO) uses regularization parameter 'alpha' to control the size of the coefficients by imposing a penalty. Please refer below tutorials for more details.
    - [Univariate Linear Regression Using Sklearn](https://satishgunjal.com/univariate_lr_scikit/)
    - [Multivariate Linear Regression Using Sklearn](https://satishgunjal.com/multivariate_lr_scikit/)
* K-fold cross validation. In this technique we divide the training data in multiple batches and use each batch for training and testing the model.
* Increasing the training data also helps to avoid overfitting.

### Example
Please refer my [Polynomial Linear Regression Fish Wgt Prediction](https://www.kaggle.com/satishgunjal/polynomial-linear-regression-fish-wgt-prediction) Kaggle notebook. In this study I am using quadratic function, to make it overfitting model you can try 10th degree function and check the results.

## Good Fitting 
* It is a sweet spot between Underfitting and Overfitting model
* A good fitting model generalizes the learnings from training data and provide accurate predictions on new data
* To get the good fitting model, keep training and testing the model till you get the minimum train and test error. Here important parameter is 'test error' because low train error may cause overfitting so always keep an eye on test error fluctuations. The sweet spot is just before the test error start to rise.

  ![Goodfit.png](https://raw.githubusercontent.com/satishgunjal/images/master/Goodfit.png)

### Example
Please refer my [Multiclass Logistic Regression](https://satishgunjal.com/multiclass_lr_sklearn/). In this study I am using Linear Model from Sklearn library to perform Multi Class Logistic Regression on handwritten digit's dataset. Notice the algorithm selection and model performance analysis.
