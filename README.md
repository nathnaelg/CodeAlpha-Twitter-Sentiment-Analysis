# 👹 Twitter Sentiment Analysis 

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

![images](images.jpg) 

## Problem Statement

The ability to analyze user sentiment through tweets and comments can provide significant value to companies during __product launches__. By understanding customer behavior and incorporating __sentiment analysis__, companies can gain insights from user feedback. This empowers them to make informed decisions, take necessary actions, and improve overall __revenue__ by addressing customer concerns and making targeted improvements accordingly.

![images](Twitter-sentiment-analysis-1.jpg)

## Machine Learning and Data Science

Our approach involves utilizing machine learning techniques and text extraction to predict the sentiment of a given text, determining whether it is positive or negative. Initially, we will analyze the text and examine the various words present within it. Once we have a comprehensive understanding of the text, we will proceed with the machine learning analysis, employing deep neural networks. The output from this analysis will be utilized in subsequent machine learning operations to generate predictions regarding the sentiment of the text, specifically determining whether it is positive or negative.

## Natural Language Processing (NLP)
We would be using the __natural language processing__ that is required when doing the machine learning analysis. Performing the natural language processing ensures that the words that are present are converted into mathematical vectors that are used for different machine learning models for prediction. Once the mathematical vectors are converted into different vectors, they are given for the machine learning models for prediction respectively. Therefore, with the features that are present in the text along with some newly created features, the machine learning, and deep learning models would be using those techniques and ensures that they are getting the best outputs respectively. 

## Vectorizers 

It is important to use vectorizers that are important for machine learning. Therefore, a given text which is in the form of a string is converted into a vectorial representation which is what is being used by machine learning models for prediction. Below are some of the vectorizers that were used in the process of converting a text into a mathematical representation. 

* [__Count Vectorizer__](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
* [__Tfidf Vectorizer__](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

## Machine Learning Models

In this project, there was only one ML model used to get the prediction of the sentiment of tweets. Below is the model that was used for the task of prediction.

* [__Deep Neural Networks__](https://www.tensorflow.org/api_docs/python/tf/keras/Model)

