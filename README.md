Twitter Sentiment Analysis
This project performs sentiment analysis on tweet data sourced from Kaggle using basic natural language processing techniques and a logistic regression model.
Dataset
The dataset used was collected from Kaggle, consisting of tweets labeled as positive or negative.
Features
Text Cleaning: Removal of non-alphabetic characters, lowercasing
Stopword Removal using NLTK
Word Stemming with Porter Stemmer
TF-IDF Vectorization
Sentiment Classification using Logistic Regression
Model Evaluation with Accuracy Score
Libraries Used
numpy
pandas
re
nltk.corpus.stopwords
nltk.stem.porter.PorterStemmer
sklearn.feature_extraction.text.TfidfVectorizer
sklearn.model_selection.train_test_split
sklearn.linear_model.LogisticRegression
sklearn.metrics.accuracy_score
