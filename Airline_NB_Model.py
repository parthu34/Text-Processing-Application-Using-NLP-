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


# TF-IDF method convert text into numeric form
df_airline_vectorizer = TfidfVectorizer (max_features=1500, min_df=12, max_df=0.95, stop_words=stopwords.words('english'))
df_airline_processed_feature = df_airline_vectorizer.fit_transform(df_airline_feature1).toarray()

# Train the model

X_train, X_test, y_train, y_test = train_test_split(df_airline_processed_feature, df_airline_sentiment, test_size=0.05, random_state = 1000)

#Naive bayes

Naive = naive_bayes.MultinomialNB()
Naive.fit(X_train,y_train)
predictions_NB = Naive.predict(X_test)
print("NB Accuracy Score: ",accuracy_score(predictions_NB,y_test)*100)