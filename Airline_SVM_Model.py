import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

# Reading datset

df_airline_clean = pd.read_csv('Airline_Tweets_PreProcessed_Cleaned_Data.csv')

#print(df_airline_clean.head())

df_airline_sentiment = df_airline_clean['airline_sentiment']
df_airline_feature1 = df_airline_clean['lemmatized_text']


# TF-IDF method for converting text into numeric form
df_airline_vectorizer = TfidfVectorizer (max_features=1000, min_df=12, max_df=0.95, stop_words=stopwords.words('english'))
df_airline_processed_feature = df_airline_vectorizer.fit_transform(df_airline_feature1).toarray()

# Training model with specified parameters
X_train, X_test, y_train, y_test = train_test_split(df_airline_processed_feature, df_airline_sentiment, test_size=0.05, random_state = 1000)


# SVM Model

Airline_SVM = svm.SVC(C=1, kernel='linear', degree=3, gamma='auto')
Airline_SVM.fit(X_train,y_train)
# predict the tweets
airline_predictions_SVM = Airline_SVM.predict(X_test)
# Printing accuracy
print("SVM Accuracy Score -> ",accuracy_score(airline_predictions_SVM, y_test)*100)
