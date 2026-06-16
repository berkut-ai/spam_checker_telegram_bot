from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import joblib

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

RANDOM_STATE = 42
TEST_SIZE = 0.2

data = pd.read_csv("data/final_train.csv")
X = data['text']
y = data['spam']

vectorizer = CountVectorizer(lowercase=True, stop_words='english')
X = vectorizer.fit_transform(X)

model = LogisticRegression(C=0.2)

model.fit(X, y)

joblib.dump((model, vectorizer), 'model.pkl')
