---
title: 'Machine Learning Introduction And Learning Plan'
date: 2020-05-01
permalink: /ml_intro/
tags:
  - Machine learning
  - Machine learning study plan
  - Beginers guide
---

![machine_learning_header.png](https://github.com/satishgunjal/images/blob/master/machine_learning_header.png?raw=true)

I this tutorial we will see the brief introduction of Machine Learning and prefferred learning plan for beginers.

## **What Is Machine Learning**

We are using machin learning every day without even noticing it. Probably you are reading this tutorial because search engine algorithm suggested it to you. Few examples of everyday machine learning uses are,

* Everytime we do Google search, result are not sorted alphabatically or datewise, a mchine learning algorihm sorts it based on relevance
* Gmail uses machine elarning algorithm to block spam email. Imagine using mail service like Gmail without spam filter?
* Google photos use machine learning algorithm for sorting you and your loved ones photos.

In so many ways, machine learning algorithms are eighter making decisions for us or helping us to make more informed decosions.

Unlike tradional fixed rule based programming machine learning is a science of getting copmuters to learn without being explicitly programmed.

## **Why Should We Learn It?**

You must have heard the buzzword **"Data Is Ne Oil"**. So if data is going to be a new oil machine learning is the best tool to make us of exabytes of data generated everyday

Any individual, organization or a country must use the machine learning tools to analize the data so that they can make well informed decision. If they failed to do it they will be left behind.

## **Learning Plan**

In my opinion best way to learn anything is to understand the concepts, practice it and then use it to solve some real world problems.

I will follow the below four step plan for eavery machine learning algorithm. 
* Step1: Understand the concepts using easy to understand intuitions 
* Step2: Write python code to implement the algorithm fromm scratch. 
* Step3: Test the algorithm with some test datastes.
* Step4: Use opensource library like scikit-learn to test same algorithm

Since scikit-learn library has all the machine learning algorithms implemented it is very tempting to just start using it. But if you are beginer, I will strongly advice you to follow above plan so that you will learn inner workings of every algorithm and it may make huge differenc ein long run.

Without further dew lets start with it...

![fresh_start.png](https://github.com/satishgunjal/images/blob/master/fresh_start.png?raw=true)

## **Defination And Types Of Machine Learning**

"A program is said to learn from experience **'E'** with respect to some task **'T'**  and some performce measure **'P'**, if its performance(P) at task(T) improves with experience(E)"  By Tom Mitchell

Machine learning is subset of Artificial intelligence. To solve machine learning problems we eighter use mathematical models or we can use neural network. Machine learning problems are of two type,
* Supervise learning
* Unsupervise learning

![machine_learning_venn.png](https://github.com/satishgunjal/images/blob/master/Machine_Learning_Venn.png?raw=true)

## **Supervised Learning**

In supervised learning our dataset contains the input values and expected output values. For example consider below dataset of house size and price

size(sqft) | price(K)
--- | ---
1200 | 3000 K
1300 | 4000 K
1400 | 5000 K

So in supervise learning we have to understand the relationship between input and output values so that we can predict the values based on input data.
Regression and classification are two types of supervised learning priblems
* **Regression problem**: If the prediction is some kind of continuous value like price of the house then it is called as regression problem

* **Classification problem**: If the prediction is some kind of classification like house will sale or not then it is called as classification problem

![regression_vs_classification.png](https://github.com/satishgunjal/images/blob/master/Regression_vs_Classification.png?raw=true)

## **Unsupervised Learning**

In case of unsupervised learning our dataset is without any set of output values or labels. Unseupervised learning is a learning setting where we give the algorithm a ton of unlabelled data and ask it to find some kind of structure in it.

Since our data is unlabelled unsupervised learning allos us to approach a problem with little or no idea what our result should look like

Clustering and association are two type sof unsupervised learning problems

* **Clustering**: Clustering problem is used to fin inherent grouping in data. e.g. In case of e-commerce finding a group of customers based on purchasing behaviour

* **Asociation**: Association problem is used to find rules that describe the large portions of data. e.g. In e-commerce finding a rule, why customers that buy X items also tends to buy Y items

![clustering_vs_association.png](https://github.com/satishgunjal/images/blob/master/clustering_vs_association.png?raw=true)

## **Machine Learning Models**

In this series we are going to learn below machine learning models

* Linear Regression
  - Univariate linear regression
  - Multivariate linear regression
* Logistic regression
  - Single classn (binary) classification
  - Multi class classification
* Decision tree
* Random Forest
* Support vector machine
* K fold crosss validation
* K means clustering
* Naive Bayes
* Hyper parameter tunning


