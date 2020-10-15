---
title: 'Introduction to NLP'
date: 2020-10-15
permalink: /intro_nlp/
tags:
  - nlp
  - text
header:
  teaser: https://raw.githubusercontent.com/satishgunjal/images/master/Header_NLP_Basics_1200x639.png
excerpt: In short NLP is an AI technique used to do text analysis. Whenever we have lots of text data to analyze we can use NLP. Apart from text analysis, NLP also used for variety of other tasks.
---

![Header_NLP_Basics_1200x639](https://raw.githubusercontent.com/satishgunjal/images/master/Header_NLP_Basics_1200x639.png)
<sub><sup>Image source https://www.forbes.com/</sup></sub>

# Index
* [Introduction](#1)
* [What is NLP?](#2)
* [Understanding the Text Data is Hard](#3)
* [NLP Workflow](#4)
  - [Text Preprocessing](#5)
    - [Noise Removal](#6)
    - [Text Normalization](#7)
    - [Object Standardization](#8)
  - [Exploratory Data Analysis](#9)
  - [Feature Engineering](#10)
  - [Model Building & Deployment](#11)
* [Libraries for NLP](#12)
* [References](#13)



# Introduction <a id ="1"></a>

We generate tons of data every day. Our WhatsApp chats, phone calls, emails, SMS's contains unstructured data which is easy for us to understand but not so easy for machines. In fact around 80% of available data is in unstructured format and considering the growth of faceless apps like Chatbot, this is going to increase. Majority of this unstructured data is in text format. It's easy for humans to analyze and process the unstructured text/audio data but it takes lots of time and quality also varies. So there is need of an automated system which can do it, that's where Natural Language Processing (NLP) technique of Artificial Intelligent(AI) comes for rescue.

So where does NLP stands in the realm of AI.

![NLP_Venn](https://raw.githubusercontent.com/satishgunjal/images/master/NLP_Venn.png)


You can see there is overlap of ML and NLP, because once we convert unstructured data to structured format we can use Ml statistical tools and algorithms to solve the problems.

# What is NLP? <a id ="2"></a>
In short NLP is an AI technique used to do text analysis. For nerds out there here is more formal definition of NLP.

**"Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data."**

So whenever we have lots of text data to analyze we can use NLP. Apart from text analysis, NLP also used for variety of other tasks. Few important use cases of NLP are,
* **Text Classification**: Using NLP we can classify given corpus of text into different groups based on label or keywords.
* **Sentiment Analysis**: To identify the sentiment(positive or negative) from given text. Very useful in case of movie/product/services reviews.
* **Relationship Extraction**: Can be used to retrieve important relationship data from text. e.g. relationship between place and person.
* **Chatbots**: NLP is one of the core building block of Chatbot platforms like Google Dialogflow, Amazon Lex.
* **Speech recognition**: NLP is used to simplify speech recognition and make it less time-consuming.
* **Question and Answering**: Using NLP we can analyze given textual data and build a model which can answer user questions.
* **Named Entity Recognition(NER)**: We can identify important information (entity) like date time, place, person etc from text using NLP.
* **Optical Character Recognition**: Given an image representing printed text, determine the corresponding text.

# Understanding the Text Data is Hard <a id ="3"></a>

Languages that we use do not follow any specific rule, consider below sentences for example.

> "Let's eat grandma.",  "kids are really sweet.",  "I'd kill for a bath."

What do you think a computer program will interpret from above sentences? Parsing any natural language input using computers is very difficult problem. Like any complex problem, in order to solve it we are going to split it into small pieces and then chain them together for final analysis. This process is called as building pipeline in machine learning terminology. Same thing we are going to do to solve Natural Language processing problems.

Before we go in details about pipeline steps let's try to understand how our text data is formatted. Our input text data can be unstructured but every sentence is collection of words and every document is collection of sentences. Every text corpus at its core is just a collection words.

![Text_Hierarchy_NLP](https://raw.githubusercontent.com/satishgunjal/images/master/Text_Hierarchy_NLP.png)

We can have text corpus of any kind of data, with one or more than one document. In case of email's we may have separate document for each email and in case of reviews we may have one single document with tab separated data for each review.

# NLP Workflow <a id ="4"></a>

Irrespective of our text data format, steps that are used to solve NLP problems remains more or less same. Major  steps that we follow while solving the NLP problems are as below.


![NLP_Workflow](https://raw.githubusercontent.com/satishgunjal/images/master/NLP_Workflow.png)

In the text preprocessing step we remove all the clutter and noise from the text. Then we perform the exploratory data analysis to understand the data. Based on our understanding from data analysis, we create new features in feature engineering step. Now once we have well formatted data with features, to create a ML model as per our requirement. In the last step we test our model and deploy it in production.

## Text Preprocessing <a id ="5"></a>
Text preprocessing is very important step in NLP workflow, without it, we can't analyze the text data. Below are the three major steps in text preprocessing.

![NLP_Text_Preprocessing](https://raw.githubusercontent.com/satishgunjal/images/master/NLP_Text_Preprocessing.png)
 
### Noise Removal <a id ="6"></a>
Any text which is not relevant to the context of the data and the task that we want to perform is considered as noise. Most common noise in text data is HTML tags, stop words, punctuations, white spaces and URL's. So in this step we remove all these noisy elements from text. Libraries such as spaCy and NLTK also has the standard dictionary of some of these noisy elements. If required we can also build our own list.

### Text Normalization <a id ="7"></a>
On higher level normalization is used to reduce the dimensions of the features so that machine learning models can efficiently process the data. Text data contains multiple representation of the same word, For example – “play”, “player”, “played”, “plays” and “playing” are the different variations of the word – “play”. These variations are useful in case of speech but not much useful for text analysis. During text normalization we convert all the disparities of a word into their normalized form. In this step we perform tokenization, lemmatization, stemming, and sentence segmentation.
* **Tokenization**: Tokenization is one of the first step in any NLP pipeline. Tokenization is nothing but splitting the raw text into small chunks of words or sentences, called tokens. For more details please refer [Tokenization in NLP](https://www.kaggle.com/satishgunjal/tokenization-in-nlp)

* **Lemmatization**: Lemmatization removes inflected ending from the word and return the base/root/dictionary form of the word. This base form of the word is knows as lemma.

* **Stemming**: It is one of the way of doing the lemmatization. Stemming involves simply lopping off easily-identified prefixes and suffixes to get the base form of the word. For example 'connect' is the base form of 'connection', here 'ion' is just suffix.

### Object Standardization <a id ="8"></a>
Text data often contains words or phrases which are not present in any standard dictionaries of spaCy or NLTK library. So we have to handle all such words with the help of custom code. In this step we fix the non-standard words with the help of regular expression and custom lookup table.

## Exploratory Data Analysis <a id ="9"></a>

![NLP_EDA](https://raw.githubusercontent.com/satishgunjal/images/master/NLP_EDA_750x500.png)

In case of unstructured text data exploratory data analysis plays an extremely important role. In this step we visualize and explore data to generate insights. Based on our understanding we try to summarize the main characteristics in data for feature generation. 

## Feature Engineering <a id ="10"></a>
In this step we convert the preprocessed data into features for machine learning models to work on. We can use below techniques to extract features from text data.

![NLP_Feature_Engineering](https://raw.githubusercontent.com/satishgunjal/images/master/NLP_Feature_Engineering.png)

<sub><sup>Image source https://www.udemy.com/</sup></sub>

* **Syntactic Parsing**: Once we have the tokens we can predict the part of speech(noun, verb, adjective etc) for it. Knowing the role of each word in the sentence will help to understand the meaning of it. We use dependency grammar and part of speech (POS) tags for syntactic analysis.
* **Entity Extraction**: It is more advanced form of language processing, that is used to identify parameter values from input text.
These parameter values can be places, people, organizations..etc. This is very useful to pickup the important topics or key section from a text input.
* **Statistical Features**: Using technique like Term Frequency-Inverse Document Frequency(TF-IDF) we can convert text data into numerical format. We can also use Word Count, Sentence Count, Punctuation Counts etc to create count/density based features.
* **Word Embedding**: Word embedding technique are used to represent the word as a vector. Popular model like Word2Vec can be used to perform such task. These word vectors can be used as features in machine learning models.

## Model Building & Deployment <a id ="11"></a>
* First step in model building is to have separate set of training and test data sets. This will make sure that our model will get tested on unknown data.
* Choose an algorithm as per task in hand. For example if we are working on classification problem then we can choose from variety of classification algorithm like Logistic Regression, Support Vector Machine, Naïve Bayes etc.
* Create pipeline which will feed the data to the model. Same pipeline then can be used to test real word data.
* Once pipeline is built test it using training dataset and evaluate the model using test dataset.
* We can use variety of metric to test the model score. Once we get satisfactory score then deploy the model in production.

# Libraries for NLP <a id ="12"></a>

* [Scikit-learn](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html): Scikit-learn is a free software machine learning library for the Python programming language. 
* [Natural Language Toolkit (NLTK)](https://www.nltk.org/): The Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and programs for symbolic and statistical natural language processing for English written in the Python programming language.
* [spaCy – Industrial-Strength Natural Language Processing.](https://spacy.io/): spaCy is an open-source software library for advanced natural language processing, written in the programming languages Python and Cython.
* [Gensim](https://pypi.org/project/gensim/): Gensim is an open-source library for unsupervised topic modeling and natural language processing, using modern statistical machine learning. Gensim is implemented in Python and Cython.
* [Stanford CoreNLP – NLP services and packages by Stanford NLP Group](https://stanfordnlp.github.io/CoreNLP/): CoreNLP enables users to derive linguistic annotations for text, including token and sentence boundaries, parts of speech, named entities, numeric and time values, dependency and constituency parses, coreference, sentiment, quote attributions, and relations.
* [TextBlob: Simplified Text Processing](https://textblob.readthedocs.io/en/dev/): TextBlob is a Python (2 and 3) library for processing textual data. It provides a simple API for diving into common natural language processing (NLP) 

# References <a id ="13"></a>
* https://medium.com/@ageitgey/natural-language-processing-is-fun-9a0bff37854e
* https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/
* https://medium.com/@suneelpatel.in/nlp-pipeline-building-an-nlp-pipeline-step-by-step-7f0576e11d08
* https://towardsdatascience.com/build-and-compare-3-models-nlp-sentiment-prediction-67320979de61
* https://towardsdatascience.com/nlp-text-preprocessing-a-practical-guide-and-template-d80874676e79
* https://towardsdatascience.com/text-preprocessing-in-natural-language-processing-using-python-6113ff5decd8#:~:text=In%20NLP%2C%20text%20preprocessing%20is,Stop%20words%20removal
