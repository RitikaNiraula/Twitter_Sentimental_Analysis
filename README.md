# 🐦 Twitter Sentiment Analysis

This project performs sentiment analysis on tweets collected from Kaggle using natural language processing (NLP) techniques and a logistic regression classifier.

---

## 📄 Dataset

The dataset consists of tweets labeled as **positive** or **negative** sentiments, sourced from Kaggle.

---

## 🧩 Features & Preprocessing Steps

- **Text Cleaning:** Removal of non-alphabetic characters and conversion to lowercase  
- **Stopword Removal:** Using NLTK's stopword list  
- **Word Stemming:** Porter Stemmer to reduce words to their root form  
- **TF-IDF Vectorization:** Converts text data into numerical features for modeling  

---

## 🤖 Model

- **Classifier:** Logistic Regression  
- **Evaluation Metric:** Accuracy Score  

---

## 🛠️ Libraries Used

- `numpy`  
- `pandas`  
- `re` (regular expressions)  
- `nltk.corpus.stopwords`  
- `nltk.stem.porter.PorterStemmer`  
- `sklearn.feature_extraction.text.TfidfVectorizer`  
- `sklearn.model_selection.train_test_split`  
- `sklearn.linear_model.LogisticRegression`  
- `sklearn.metrics.accuracy_score`  

---

