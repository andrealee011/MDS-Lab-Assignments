# Assignment from DSCI 571 Lab 2
# Using naive bayes to perform sentiment analysis on IMDB movie reviews

# Import libraries
import pandas as pd
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Load data (dataset from https://www.kaggle.com/utathya/imdb-review-dataset)
imdb = pd.read_csv("data/imdb_master.csv", encoding="latin_1", index_col = 0)
imdb_df = imdb.query("label != 'unsup'")

# 2. Split data
X = imdb_df['review']
y = imdb_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y.to_numpy().ravel(), test_size = 0.2)

# 3. Preprocess data
tv = TfidfVectorizer()
X_train_tfidf = tv.fit_transform(X_train)
X_test_tfidf = tv.transform(X_test)

# 4. Fit model
nb_tfidf = MultinomialNB(alpha=1)
nb_tfidf.fit(X_train_tfidf, y_train)
nb_tfidf_train_error = 1- nb_tfidf.score(X_train_tfidf, y_train)
nb_tfidf_test_error = 1- nb_tfidf.score(X_test_tfidf, y_test)

# 5. Calculate error
print('The training error for the tf-idf representation is %.2f.' % nb_tfidf_train_error)
print('The testing error for the tf-idf representation is %.2f.' % nb_tfidf_test_error)

# 6. Test model on fake reviews
fake_reviews = ['This movie was excellent! The performances were oscar-worthy!',
               'Unbelievably disappointing.', 
               'Full of zany characters and richly applied satire, and some great plot twists',
               'This is the greatest screwball comedy ever filmed',
               'It was pathetic. The worst part about it was the boxing scenes.', 
               '''It could have been a great movie. It could have been excellent, 
                and to all the people who have forgotten about the older, 
                greater movies before it, will think that as well. 
                It does have beautiful scenery, some of the best since Lord of the Rings. 
                The acting is well done, and I really liked the son of the leader of the Samurai.
                He was a likeable chap, and I hated to see him die...
                But, other than all that, this movie is nothing more than hidden rip-offs.
                '''
              ]
true_labels = ['pos', 'neg', 'pos', 'pos', 'neg', 'neg']

data = []
j = 0
for i in fake_reviews:
    X_test = cv.transform([i]).toarray()
    prob = nb_tfidf.predict_proba(X_test)
    neg_prob = round(prob[0][0],2)
    pos_prob = round(prob[0][1],2)
    if neg_prob > pos_prob:
        pred = 'neg'
    elif neg_prob < pos_prob:
        pred = 'pos'
    data.append([i, neg_prob, pos_prob, pred, true_labels[j]])
    j += 1
pd.DataFrame(data, columns = ['review', nb.classes_[0], nb.classes_[1], 'predicted sentiment', 'true sentiment'])
