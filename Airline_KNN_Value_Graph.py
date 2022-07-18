import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Reading datset

df_airline_clean = pd.read_csv('Airline_Tweets_PreProcessed_Cleaned_Data.csv')

#print(df_airline_clean.head())

df_airline_sentiment = df_airline_clean['airline_sentiment']
df_airline_feature1 = df_airline_clean['lemmatized_text']


# TF-IDF method convert text into numeric form
df_airline_vectorizer = TfidfVectorizer (max_features=1500, min_df=12, max_df=0.95, stop_words=stopwords.words('english'))
df_airline_processed_feature = df_airline_vectorizer.fit_transform(df_airline_feature1).toarray()

# Train the model

X_train, X_test, y_train, y_test = train_test_split(df_airline_processed_feature, df_airline_sentiment, test_size=0.05, random_state = 1000)

#Reference: https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
# Using K values from 1 to 40, we can plot a graph that shows minimum error rate
error = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

plt.show()