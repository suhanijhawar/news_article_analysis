#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
nltk.download('omw-1.4')

# Importing the dataset
df = pd.read_excel(r'C:\Users\Suhani PC\Documents\Pgm\New folder\Assignment1.xlsx')


# In[25]:


print(df.columns)


# In[26]:


# Step 1: Cleanup Articles by Preprocessing
df["Sentiment"] = df["Sentiment"].astype("str")
def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text.lower())
    # Remove punctuation
    tokens = [word for word in tokens if word not in punctuation]
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

df['cleaned_text'] = df['Article'].apply(preprocess_text)


# In[27]:


# Step 2: Checking the Mood by classification
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['Sentiment'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=1000)  # Adjusting max_features as needed
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)
y_pred = svm_classifier.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Mood Classification Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))


# In[30]:


# Step 3: Topic modeling (finding connections)
lda_model = LatentDirichletAllocation(n_components=3, random_state=42)
X_topics = lda_model.fit_transform(X_train_tfidf)
print("Topics in LDA model:")
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda_model.components_):
    print(f"Topic #{topic_idx}:")
    print(" ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]))


# In[31]:


# Step 4(optional): Aspect analysis
aspects = ['drug', 'food', 'company', 'law', 'date']  # Example aspects
for aspect in aspects:
    aspect_sentiment = []
    for article in df['cleaned_text']:
        if aspect in article:
            # Perform sentiment analysis for the aspect
            sentiment = 'positive'  # Example sentiment
            aspect_sentiment.append(sentiment)
        else:
            aspect_sentiment.append(None)
    df[aspect + '_sentiment'] = aspect_sentiment


# In[32]:


# Output summaries, mood ratings, and aspect sentiments (optional)
for index, row in df.iterrows():
    print(f"Article #{index}:")
    print("Summary:", row['Summary'])
    print("Mood:", row['Sentiment'])
    for aspect in aspects:
        print(f"{aspect.capitalize()} Sentiment:", row[aspect + '_sentiment'])
    print()


# In[33]:


print("Project by Suhani Jhawar")


# In[ ]:




