---
title: 'Program with results'
date: 2020-05-01
permalink: /posts/2012/08/blog-post-1/
tags:
  - cool posts
  - category1
  - category2
---


# Python Code



## Notations used
* m   = no of training examples (no of rows of feature matrix)
* n   = no of features (no of columns of feature matrix)
* x's = input variables / independent variables / features
* y's = output variables / dependent variables / target

## Import the required libraries
* numpy : Numpy is the core library for scientific computing in Python. It is used for working with arrays and matrices.
* pandas: Used for data manupulation and analysis
* matplotlib : Its plotting library and we are going to use it for data visualization

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Load the data
* We are going to use 'profits_and_populations_from_the_cities.csv' csv file
* File contains two columns, the frst column is the population of a city and the second column is the profit of a food truck in that city. A negative value for profit indicates a loss.

```
df = pd.read_csv('https://raw.githubusercontent.com/satishgunjal/Datasets/master/profits_and_populations_from_the_cities.csv')
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


