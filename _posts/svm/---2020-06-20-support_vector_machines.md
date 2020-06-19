---
title: 'Support Vector Machines'
date: 2020-06-20
permalink: /svm/
tags:
  - Classification
  - Sklearn
header:
  teaser: https://raw.githubusercontent.com/satishgunjal/images/master/SVM_Header.png
excerpt: Support vector machines is one of the most powerful 'Black Box' machine learning algorithm. It belongs to the family of supervised learning algorithm. Used to solve classification as well as regression problems. Tutorial includes classification example using Python 3 environment and sklearn library.

---

![SVM_Header](https://raw.githubusercontent.com/satishgunjal/images/master/SVM_Header.png)

Support vector machines (SVM) is one of the most powerful 'Black Box' machine learning algorithm. It belongs to the family of supervised learning algorithm. Used to solve classification as well as regression problems. Using threshold to classify the different groups is not very accurate and may lead to the wrong predictions. SVM classifier also use the threshold to classify the data, but it uses midpoint of the observations on the edge of each group as threshold. In order to avoid the classification error, SVM also defines the safety margin on both the side of threshold. So this safety margin will provide robustness to SVM and that's why it is called as 'Large Margin Classifier'. SVM algorithm can be used to classify linear as well as non-linear data. So the secret sauce of the SVM is the way it finds the threshold and define the safety margin on either side of threshold to avoid the classification errors.

# How SVM Classifier Work
Outliers in the data can affect the threshold value and lead to wrong predictions. Consider below example

![Datapoint_With_Outlier_And_Threshold](https://raw.githubusercontent.com/satishgunjal/images/master/Datapoint_With_Outlier_And_Threshold.png)

So from above example its clear that we can't just choose data point on edge to draw a decision boundary. In order to avoid this, SVM use cross validation technique to identify the data points to draw the decision boundary. The data points on the edge and within the boundary are called support vectors. Data points inside the margin are also called as 'misclassified observations'. So SVM using cross validation tries multiple combination of support vectors to find the best decision boundary and finally select the best possible support vector which provides the larger margin. Consider below example where SVM has ignored the few observations in order to find more robust threshold and safety margin.

![Datapoint_With_Outlier_And_SVM_Classifier](https://raw.githubusercontent.com/satishgunjal/images/master/Datapoint_With_Outlier_And_SVM_Classifier.png)



## Types Of SVM Classifiers
Since SVM can be used to classify linear as well as non-linear data, we can have support vector classifiers from a point to hyperplane. Hyperplane in this context is tool that separates the data space into one less dimension for easier classification. Actually every SVM classifier is hyperplane of dimension n - 1 where, n is the dimension of given data. 

For 1 dimensional data, hyperplane is a point, and we can classify the new observation based on the which side of the point they are

![1D_SVM_Classifier](https://raw.githubusercontent.com/satishgunjal/images/master/1D_SVM_Classifier.png)

For 2 dimensional data, hyperplane is a line, and we can classify the new observation based on the which side of the line they are

![2D_SVM_Classifier](https://raw.githubusercontent.com/satishgunjal/images/master/2D_SVM_Classifier.png)

For 3 dimensional data, hyperplane is a plane, and we can classify the new observation based on the which side of the plane they are

![3D_SVM_Classifier](https://raw.githubusercontent.com/satishgunjal/images/master/3D_SVM_Classifier.png)

For n dimensional data, hyperplane is  n-1 dimensional plane

Since we can visualize the data up to 3 dimensions, we refer the hyperplane by more friendly names like line, plane etc and for more than 3 dimensions we just call it hyperplane.

# Kernels
Kernel is the technique used by SVM to classify the non-linear data. Kernel functions are used to increase the dimension of the data, so that SVM can fit the optimum hyperplane to separate the data. Consider below example where 1D data points are randomly grouped. Using kernel function(here 2nd degree polynomial) we can convert 1D data points to 2D data points and fit a line to separate the data into two groups.

![SVM_Polynomial_Kernel](https://raw.githubusercontent.com/satishgunjal/images/master/SVM_Polynomial_Kernel.png)

So kernel functions help SVM to transform the lower dimension data to higher dimension but in order to figure out which kernel function to use, to transform the data SVM uses cross validation. Using cross validation SVM tries multiple combination of Kernel functions like polynomial kernel and choose the one which results in the best classification.

# Advantages
* Effective in high dimensional input data
* Effective when number of dimensions are higher than number of samples
* Uses a subset of training points to find the support vector classifier, so it is also memory efficient
* Prediction accuracy is higher when there is a clear separation between classes

# Disadvantages
* Large data sets- takes lots of time to separate the data
* Data with lots of error- Since SVM separates the two groups of data based on nearest points, if these points have errors in them, then it will affect entire model performance
* Choose the wrong kernel- If we choose the wrong separation plane then it will affect the model performance

# Example: Classification Problem
Now we will implement the SVM algorithm using sklearn library and build a classification model that estimates an applicant’s probability of admission based on Exam 1 and Exam 2 scores. Note- I have also used the same dataset in [Logistic Regression From Scratch With Python](https://satishgunjal.github.io/binary_lr/)

## Import Libraries
* pandas: Used for data manipulation and analysis
* numpy : Numpy is the core library for scientific computing in Python. It is used for working with arrays and matrices.
* matplotlib : It’s plotting library, and we are going to use it for data visualization
* model_selection: Here we are going to use model_selection.train_test_split() for splitting the data
* svm: Sklearn support vector machine model


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import svm
```

## Load Data
* We are going to use ‘admission_basedon_exam_scores.csv’ CSV file
* File contains three columns Exam 1 marks, Exam 2 marks and Admission status


```python
df = pd.read_csv('https://raw.githubusercontent.com/satishgunjal/datasets/master/admission_basedon_exam_scores.csv')
print('Shape of data= ', df.shape)
df.head()
```

    Shape of data=  (100, 3)
    




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
      <th>Exam 1 marks</th>
      <th>Exam 2 marks</th>
      <th>Admission status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34.623660</td>
      <td>78.024693</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30.286711</td>
      <td>43.894998</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35.847409</td>
      <td>72.902198</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60.182599</td>
      <td>86.308552</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>79.032736</td>
      <td>75.344376</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Data Understanding
* There are total 100 training examples (m= 100 or 100 no of rows)
* There are two features Exam 1 marks and Exam 2 marks
* Label column contains application status. Where ‘1’ means admitted and ‘0’ means not admitted

### Data Visualization
To plot the data of admitted and not admitted applicants, we need to first create separate data frame for each class(admitted/not-admitted)


```python
df_admitted = df[df['Admission status'] == 1]
print('Training examples with admission status 1 are = ', df_admitted.shape[0])
df_admitted.head(3)
```

    Training examples with admission status 1 are =  60
    




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
      <th>Exam 1 marks</th>
      <th>Exam 2 marks</th>
      <th>Admission status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>60.182599</td>
      <td>86.308552</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>79.032736</td>
      <td>75.344376</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>61.106665</td>
      <td>96.511426</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_notadmitted = df[df['Admission status'] == 0]
print('Training examples with admission status 0 are = ', df_notadmitted.shape[0])
df_notadmitted.head(3)
```

    Training examples with admission status 0 are =  40
    




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
      <th>Exam 1 marks</th>
      <th>Exam 2 marks</th>
      <th>Admission status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34.623660</td>
      <td>78.024693</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30.286711</td>
      <td>43.894998</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35.847409</td>
      <td>72.902198</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Now lets plot the scatter plot for admitted and not admitted students


```python
def plot_data(title):    
    plt.figure(figsize=(10,6))
    plt.scatter(df_admitted['Exam 1 marks'], df_admitted['Exam 2 marks'], color= 'green', label= 'Admitted Applicants')
    plt.scatter(df_notadmitted['Exam 1 marks'], df_notadmitted['Exam 2 marks'], color= 'red', label= 'Not Admitted Applicants')
    plt.xlabel('Exam 1 Marks')
    plt.ylabel('Exam 2 Marks')
    plt.title(title)
    plt.legend()
 
plot_data(title = 'Admitted Vs Not Admitted Applicants')
```


![Admitted_Vs_NotAdmitted_Applicants](https://raw.githubusercontent.com/satishgunjal/images/master/Admitted_Vs_NotAdmitted_Applicants.png)


## Build Machine Learning Model


```python
#Lets create feature matrix X and label vector y
X = df[['Exam 1 marks', 'Exam 2 marks']]
y = df['Admission status']

print('Shape of X= ', X.shape)
print('Shape of y= ', y.shape)
```

    Shape of X=  (100, 2)
    Shape of y=  (100,)
    

### Create Test And Train Dataset
* We will split the dataset, so that we can use one set of data for training the model and one set of data for testing the model
* We will keep 20% of data for testing and 80% of data for training the model
* If you want to learn more about it, please refer [Train Test Split tutorial](https://satishgunjal.com/train_test_split/)


```python
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state= 1)

print('X_train dimension= ', X_train.shape)
print('X_test dimension= ', X_test.shape)
print('y_train dimension= ', y_train.shape)
print('y_train dimension= ', y_test.shape)
```

    X_train dimension=  (80, 2)
    X_test dimension=  (20, 2)
    y_train dimension=  (80,)
    y_train dimension=  (20,)
    

Now lets train the model using SVM classifier


```python
# Note here we are using default SVC parameters
clf = svm.SVC()
clf.fit(X_train, y_train)
print('Model score using default parameters is = ', clf.score(X_test, y_test))
```

    Model score using default parameters is =  0.85
    

In order to visualize the results better lets create a function to plot SVM Classifier decision boundary with margin


```python
def plot_support_vector(classifier):
    """
    To plot decsion boundary and margin. Code taken from Sklearn documentation.

    I/P
    ----------
    classifier : SVC object for each type of kernel

    O/P
    -------
    Plot
    
    """
    clf =classifier
    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')  
```


```python
plot_data(title = 'SVM Classifier With Default Parameters')
plot_support_vector(clf)  
```


![SVM_Classifier_With_Default_Params](https://raw.githubusercontent.com/satishgunjal/images/master/SVM_Classifier_With_Default_Params.png)




## SVM Parameters
* Gamma: In case of high value of Gamma decision boundary is dependent on observations close to it, where in case of low value of Gamma, SVM will consider the far away points also while deciding the decision boundary
* Regularization parameter(C): Large C will result in overfitting and which will lead to lower bias and high variance. Small C will result in underfitting and which will lead to higher bias and low variance. For more details about it please refer [Underfitting & Overfitting](https://satishgunjal.github.io/underfitting_overfitting/)

So regularization parameter C and gamma parameters plays an important role in order to find the best fit model. Let's create a function which will try multiple such values and return the best value of C and gamma for our choice of the kernel. At the end we will plot the decision boundary with margin using the best choice of SVM parameters for each type of kernel.


```python
def svm_params(X_train, y_train, X_test, y_test):
    """
    Finds the best choice of Regularization parameter (C) and gamma for given choice of kernel and returns the SVC object for each type of kernel

    I/P
    ----------
    X_train : ndarray
        Training samples
    y_train : ndarray
        Labels for training set
    X_test : ndarray
        Test data samples
    y_test : ndarray
        Labels for test set.

    O/P
    -------
    classifiers : SVC object for each type of kernel
    
    """
    C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 40]
    gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 40]
    kernel_types = ['linear', 'poly', 'rbf']
    classifiers = {}
    max_score = -1
    C_final = -1
    gamma_final = -1
    for kernel in kernel_types:                    
        for C in C_values:
            for gamma in gamma_values:
                clf = svm.SVC(C=C, kernel= kernel, gamma=gamma)
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                #print('C = %s, gamma= %s, score= %s' %(C, gamma, score))
                if score > max_score:
                    max_score = score
                    C_final = C
                    gamma_final = gamma
                    classifiers[kernel] = clf        
        print('kernel = %s, C = %s, gamma = %s, score = %s' %(kernel, C_final, gamma_final, max_score))
    return classifiers
```

Lets call the svm_params() function to get the best parameters for each type of kernel


```python
classifiers = svm_params(X_train, y_train, X_test, y_test)
```

    kernel = linear, C = 0.01, gamma = 0.01, score = 0.85
    kernel = poly, C = 0.01, gamma = 0.01, score = 0.95
    kernel = rbf, C = 1, gamma = 0.03, score = 1.0
    


```python
plot_data(title = 'SVM Classifier With Parameters ' + str(classifiers['linear']))
plot_support_vector(classifiers['linear'])
```

![SVM_Classifier_C_0.01_Gamma_0.01_Kernel_Linear](https://raw.githubusercontent.com/satishgunjal/images/master/SVM_Classifier_C_0.01_Gamma_0.01_Kernel_Linear.png)



```python
plot_data(title = 'SVM Classifier With Parameters ' + str(classifiers['rbf']))
plot_support_vector(classifiers['rbf'])
```


![SVM_Classifier_C_1_Gamma_0.03_Kernel_Rbf](https://raw.githubusercontent.com/satishgunjal/images/master/SVM_Classifier_C_1_Gamma_0.03_Kernel_Rbf.png)



```python
plot_data(title = 'SVM Classifier With Parameters ' + str(classifiers['poly']))
plot_support_vector(classifiers['poly'])
```


![SVM_Classifier_C_0.01_Gamma_0.01_Kernel_Poly](https://raw.githubusercontent.com/satishgunjal/images/master/SVM_Classifier_C_0.01_Gamma_0.01_Kernel_Poly.png)


# Conclusion
Remember that, our data is 2D so hyperplane will be a line. But if you observe the data closely there is no clear separation between classes that's why straight line is not a good fit, which is obvious from above plots. Though the accuracy of poly kernel is less than rbf, but still its best choice for our data. 
