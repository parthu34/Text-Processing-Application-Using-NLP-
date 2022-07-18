import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


# Reading clean datset

df_airline_clean = pd.read_csv('Airline_Tweets_PreProcessed_Cleaned_Data.csv')

# Selecting centroids for the model
df_airline_sentiment = df_airline_clean['airline_sentiment']
df_airline_feature1 = df_airline_clean['lemmatized_text']


# TF-IDF method for converting text into numeric form
df_airline_vectorizer = TfidfVectorizer (max_features=1500, min_df=12, max_df=0.95, stop_words=stopwords.words('english'))
df_airline_processed_feature = df_airline_vectorizer.fit_transform(df_airline_feature1).toarray()

# Training model with specified parameters

X_train, X_test, y_train, y_test = train_test_split(df_airline_processed_feature, df_airline_sentiment, test_size=0.05, random_state = 1000)

# KNN Model

airline_scaler = StandardScaler()
airline_scaler.fit(X_train)

X_train = airline_scaler.transform(X_train)
X_test = airline_scaler.transform(X_test)

airline_classifier = KNeighborsClassifier(n_neighbors=4)
airline_classifier.fit(X_train, y_train)

y_pred = airline_classifier.predict(X_test)

# Printing final accuracy and predictions
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy Score by KNN Model: ",accuracy_score(y_pred, y_test)*100)
