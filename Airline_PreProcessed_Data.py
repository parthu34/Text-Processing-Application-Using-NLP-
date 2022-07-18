import nltk
import pandas as pd
import string
import re

# All stopwords available in English
allStopwords = nltk.corpus.stopwords.words('english')

ps = nltk.PorterStemmer()

wn = nltk.WordNetLemmatizer()

# Reading datset

df_airline = pd.read_csv('Tweets.csv')


#Function to remove Punctuation
def remove_punct(tweet):
    nonpunctText = "".join([char for char in tweet if char not in string.punctuation])
    return nonpunctText

# Storing into new column
df_airline['text_without_punct'] = df_airline['text'].apply(lambda x: remove_punct(x))


# Function to Tokenize words
def tokenize(tweet):
    tokens = re.split('\W+', tweet)
    return tokens

df_airline['tokenized_text'] = df_airline['text_without_punct'].apply(lambda x: nltk.tokenize.WordPunctTokenizer().tokenize(x.lower()))
df_airline['tokenized_text'] = df_airline['text_without_punct'].apply(lambda x: tokenize(x.lower()))

# Function to remove Stopwords
def remove_stopwords(listOfTokens):
    text = [word for word in listOfTokens if word not in allStopwords]
    return text

# Storing into new column
df_airline['text_without_stopwords'] = df_airline['tokenized_text'].apply(lambda x: remove_stopwords(x))

# Function for stemming

def stemming(textForStemming):
    text = [ps.stem(word) for word in textForStemming]
    return text

# Storing into new column
df_airline['stemmed_text'] = df_airline['text_without_stopwords'].apply(lambda x: stemming(x))

# Function for lemmatizing

def lemmatizing(textForLemmatizing):
    text = [wn.lemmatize(word) for word in textForLemmatizing]
    return text

# Storing into new column
df_airline['lemmatized_text'] = df_airline['text_without_stopwords'].apply(lambda x: lemmatizing(x))

print(df_airline.head())

# Saving cleaned data into CSV file

df_airline.to_csv("Airline_Tweets_PreProcessed_Cleaned_Data.csv", sep=',')

