import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import  ENGLISH_STOP_WORDS, TfidfVectorizer

df=pd.read_csv('dataset/twitter_sentiments.csv')

train, test = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=21)

vector = TfidfVectorizer(lowercase=True, max_features=1000, stop_words=ENGLISH_STOP_WORDS)
vector.fit(train.tweet)
train_idf = vector.transform(train.tweet)
test_idf = vector.transform(test.tweet)

model = LogisticRegression()
model.fit(train_idf, train.label)
predict_train = model.predict(train_idf)
predict_test = model.predict(test_idf)

f1_score(y_true=train.label, y_pred=predict_train)
f1_score(y_true=test.label, y_pred=predict_test)

pipeline = Pipeline(steps = [('tfidf', TfidfVectorizer(lowercase=True, max_features=1000, stop_words=ENGLISH_STOP_WORDS)),('model', LogisticRegression())])
pipeline.fit(train.tweet, train.label)

from joblib import dump
dump(pipeline, filename="my_ML.joblib")