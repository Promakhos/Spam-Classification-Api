import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from pathlib import Path

class SpamClassifierService:
    def __init__(self,):
        self.cv = CountVectorizer()
        self.clf = MultinomialNB()
        self.score = None
        

    def train_model(self, data_path):
        data = pd.read_csv(data_path)
        x = np.array(data["Message"])
        y = np.array(data["Category"])
        X = self.cv.fit_transform(x)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        self.clf.fit(X_train, y_train)
        self.score = self.clf.score(X_train, y_train)

    def get_score(self):
        return self.score

    def predict_spam(self, message):
        email = self.cv.transform([message]).toarray()
        prediction = self.clf.predict(email)[0]
        return prediction

data_file_path = Path(__file__).parent / "spam.csv"