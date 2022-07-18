# Text-Processing-Application-Using-NLP-
Airline User Review Tweets – A Sentiment Analysis


Introduction

This project focuses on a text-processing application that uses Natural Language Processing (NLP) to read a dataset that is specific to a business and serves the purpose of enhancing or improvement of a feature helpful for the growth of the business.
In general, if one needs to define NLP, it is a branch of Artificial Intelligence that deals with the interaction between Computer and Humans using the natural language as a medium to analyze, manipulate as well as generate. It has a variety of uses such as Auto-Correction, Auto-fill, Spam filter, Text processing, Sentiment Analysis and much more.
Problem Statement

Here, we have taken the Airline Industry as our business domain. We are targeting 6 Major US Airlines and analyzing the reviews made by the customers after their flight or other experience with any of those companies. The companies whose reviews data is analyzed are :
1.	US Airways
2.	United Airlines
3.	American Airlines
4.	Southwest
5.	Virgin America
6.	Delta

Reviews need to be analyzed into 3 Categories – Positive, Negative and Neutral.
 
Solution

This will be a supervised machine learning task where we process the text and put it under pre-defined categories. Specifically, we will be doing sentiment analysis on the user data. We will be following the common machine learning pipeline for this which is discussed further below.
Choosing a dataset is the most important task in the Machine Learning task and we have chosen a dataset contains a substantial amount of data containing the user reviews as tweets addressed to one of the Airline Company mentioned above.
Our text processing application will use that data to differentiate reviews into 3 categories – Positive, Negative and Neutral.
Naked eye data analysis of the dataset showed the following results :

1.	The data is too ambiguous and needs to be converted to a uniform type
2.	The dataset is too large to be interpreted theoretically, thus we will use graphical analysis to visualize data (exploratory data analysis)
Following the general rules of developing a model, we will first clean the data set and do the NLP tasks starting with Syntax Analysis (Grammar Induction, Lemmatization, Parsing, Stemming, etc.)
As a team effort, we will be using 5 Models (which will be revealed further in the document). These steps To elaborate on these steps, we can list them as follows:
1.	Exploratory data analysis
2.	Dataset Cleaning
3.	Pre-Process the data
4.	Tf- IDF
5.	Divide the datasets – Train set and Test set
6.	Training the Model
7.	Improve the performance of the models by tuning the Features

Before we start with the above-mentioned steps, we need to configure our machine and the Python IDE with the required libraries and packages.
 
Import the Packages

The common packages which will be required for this solution are:

1.	NLTK – NLP Library in Python. We will use state_union, stopwords, and wordnet in specific
2.	Scikit-Learn – Python Machine Learning Library
3.	Pandas – Data Manipulation and Analysis
4.	Re – String searching and Manipulation 

NLTK Library:

![image](https://user-images.githubusercontent.com/29891369/179540161-32372679-0f13-4272-91b1-f7ef37a9ed82.png)

 
However, there are some packages specific to some models which are as follows:

Logistic Regression:
![image](https://user-images.githubusercontent.com/29891369/179540234-2dd714d0-a639-4dd4-b410-26f51158e7b8.png)


Random Forest Classifier:
![image](https://user-images.githubusercontent.com/29891369/179540256-fbea1c84-79a6-4b50-a15b-8718c75fd07d.png)


K – Nearest Neighbour:
![image](https://user-images.githubusercontent.com/29891369/179540276-40a99350-ae25-4e52-b9b0-bd1a3c7d1aeb.png)


Naive Bayesian Classifier:
![image](https://user-images.githubusercontent.com/29891369/179540297-9d73aafc-9519-442f-87b2-de099615777f.png)


Support Vector Machine:
![image](https://user-images.githubusercontent.com/29891369/179540322-5cb83ca7-de40-4a51-a596-cfbb3a37fd8e.png)

 
Import the Dataset

We have downloaded the dataset from the following link and the data is located on our local computer.
And we will import the data using the following code:

The source URL is: https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US- Airline-Sentiment-/master/Tweets.csv

![image](https://user-images.githubusercontent.com/29891369/179540374-ac7bc474-069b-4f7a-9df9-556ff9552be2.png)



Exploratory Data Analysis

We would like to know the percentage of each review in the dataset – Positive, Negative and Neutral
Graph 1
![image](https://user-images.githubusercontent.com/29891369/179540395-dff614e2-4f37-460a-b657-4e9ea52165b8.png)

 
Then, we would like to know the Total number of tweet reviews for each Airline

Graph 2
![image](https://user-images.githubusercontent.com/29891369/179540442-fbf6f20b-e987-498b-83ad-e9b87784bad0.png)




Finally, let us have a look at the reviews divided categorically for each of those 6 Airlines
Graph 3
![image](https://user-images.githubusercontent.com/29891369/179540464-a9747d1d-67bf-4a75-b13d-2d503f361aef.png)



 
Data Cleaning

The datasets contain the tweets by multiple users which makes the dataset have slangs, punctuation marks and much more. Thus we need to do the following task to clean and prepare our data:
1.	Remove Punctuation Marks
![image](https://user-images.githubusercontent.com/29891369/179540499-8077a3e6-3b60-4b42-9239-9bee63d32376.png)


2.	Tokenize the data – Separate the tweets into tokens (Tokens may be sentences or words)
![image](https://user-images.githubusercontent.com/29891369/179540620-dc96ceec-ebb4-43d3-bd5a-36eae519b2dc.png)

 
3.	Elimination of stopwords – Remove the commonly occurring words which convey almost no information about the data
![image](https://user-images.githubusercontent.com/29891369/179540658-1554104a-6b04-4134-9d67-4f7ace8d557a.png)
 
 
Pre-Process the Data

Now we begin the pre-processing of the data by using NLP Syntax evaluation tasks

1.	Stemming the data – Treat same words with different forms of tense, verb, and grammar as same words (reduces the corpus of the words) – word back to its root form
![image](https://user-images.githubusercontent.com/29891369/179540724-b6e02888-ce2c-4395-bd03-98979b10889b.png)


2.	Lemmatizing the data – This helps us derive the Lemma – aka the canonical form of the word – base dictionary for of a word
![image](https://user-images.githubusercontent.com/29891369/179540757-d2576b36-a59b-4adc-9358-c4cf85eab906.png)


3.	Now we will save the pre-processed data to a new file.
![image](https://user-images.githubusercontent.com/29891369/179540771-391f7f4c-9678-4701-89c7-5284098a440c.png)

 
Vectorizing the Data using TF- IDF

TF = Term Frequency

IDF = Inverse Document Frequency

It improves the contribution of words towards classification in the whole dictionary while its occurrence is less as compared to a particular document (here it is our CSV file) It helps us find the word in the document as well as will fetch the word from a dictionary and gives binary values (0 or 1) as output and then store it to an array. Often it is used to plot the relative frequency of the word. Generally, it is used for improving the search engine score. (This is implemented in each Model and not on the whole dataset at once)
Here, the SciKit has TfIDVectorizer which allows us to convert the text to TF-IDF
![image](https://user-images.githubusercontent.com/29891369/179540805-7fcfa492-d77a-47f4-9d6d-7d565fd0b276.png)


The variables used are :

Max-features = 1500 – 1500 Most frequently occurring words

Max_df = 0.95 - 95% of the most frequently occurring words in the dataset.

Min_df = 12 – At least 12 of the documents in the dictionary has these particular words
These variables are kept constant for all the models which we will train on the Airline Tweets dataset.
Divide the dataset - Train set and Test set

We will divide the dataset into parts Train set – 95% of the dataset
Test set – 5 % of the dataset
![image](https://user-images.githubusercontent.com/29891369/179540835-97cad2c3-eaf2-4ce5-bb12-b56d700f29cf.png)

 
Training the Model

Logistic Regression:
![image](https://user-images.githubusercontent.com/29891369/179540863-2cae27d8-f66b-48c7-88a2-30dbdd2192d6.png)


Random Forest Classifier:
![image](https://user-images.githubusercontent.com/29891369/179540884-05037c1d-c2ff-4218-a5ef-c70c66b8edb3.png)


K – Nearest Neighbour:
![image](https://user-images.githubusercontent.com/29891369/179540912-978524a5-0db4-4eca-a409-b828e588cb19.png)

 
Now for K-values, first we randomly chose the value of K and then we tried to predict it by plotting the error rate for the K -Values for (1-40) to take the best K-value into consideration
![image](https://user-images.githubusercontent.com/29891369/179540953-03a73f33-7eea-4672-9a6c-ef60c57f3565.png)


The graph looks like this:
![image](https://user-images.githubusercontent.com/29891369/179540995-c8a1c6ee-84c6-41ff-b891-0af05a3f6b16.png)




The lowest error rate was for value K = 4 from the graph, thus that is used in the above-shown code to train the model
 
Naive Bayesian Classifier:
![image](https://user-images.githubusercontent.com/29891369/179541037-1cf0bdc4-2e20-4223-afef-b0ab1ee0ae2a.png)


Support Vector Machine:
![image](https://user-images.githubusercontent.com/29891369/179541059-14072502-287f-4e84-acb2-ca6fa1fbf1da.png)

 
Output and Accuracy of the Models

Logistic Regression:
![image](https://user-images.githubusercontent.com/29891369/179541108-693916b8-9f17-444d-9742-cfccc172862f.png)


Random Forest Classifier:
![image](https://user-images.githubusercontent.com/29891369/179541144-4127bea5-13e7-49a8-acf3-0e4b412c8a14.png)


K – Nearest Neighbour:
![image](https://user-images.githubusercontent.com/29891369/179541212-ece9cc19-5569-4877-b059-da9853178722.png)


Naive Bayesian Classifier:
![image](https://user-images.githubusercontent.com/29891369/179541233-7a00992f-8de4-48e2-86a6-44c5243b01a5.png)

 
Support Vector Machine:
![image](https://user-images.githubusercontent.com/29891369/179541255-54834b2f-5f4b-4875-9e5f-d91cb8f10cdf.png)


Comparison of the Accuracy of All Models
![image](https://user-images.githubusercontent.com/29891369/179541764-9968afcd-a94a-4bc1-b6ac-104acfa58ac9.png)


Conclusion
We used 5 Models to classify the user tweets into 3 pre-classified categories – Positive, Negative and Neutral. Out of the 5 Models which we trained on 95% of our dataset the best fit model for the Airline Tweet Review Sentiment Analysis is the Logistic Regression Model with an accuracy of 81.42 %.
This project helped us to gain insights about Sentiment analysis using different Python libraries and training 5 Models on a pretty big dataset.

