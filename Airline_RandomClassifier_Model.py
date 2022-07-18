import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
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


# Random Classifier

airline_classifier = RandomForestClassifier(n_estimators=500, random_state=0)
airline_classifier.fit(X_train, y_train)

# Evaluate the model

airlien_predictions = airline_classifier.predict(X_test)

print(confusion_matrix(y_test,airlien_predictions))
print(classification_report(y_test,airlien_predictions))
print("Accuracy from Random Classifier model: ",accuracy_score(y_test, airlien_predictions)*100)