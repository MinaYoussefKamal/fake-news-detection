import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
import spacy
import joblib
import re
nlp = spacy.load("en_core_web_sm")

fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = 0
true_df["label"] = 1



data = pd.concat([fake_df,true_df],ignore_index=True)

data["content"] = data["title"] + " " + data["text"]

data = data.sample(frac= 1).reset_index(drop=True)



stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()  # lowercase the string
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'(<br\s*/?>|br\s*/?|/br)', '', text)
    doc = nlp(text)
    text = " ".join([token.lemma_ for token in doc if token.text not in stop_words])
    return text

data["content"] = data["content"].astype(str).apply(preprocess)


x_train, x_test, y_train, y_test = train_test_split(data["content"],data["label"], test_size=0.2,random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

model = LinearSVC()
model.fit(x_train_vec, y_train)

y_pred = model.predict(x_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", classification_report(y_test, y_pred))

joblib.dump(vectorizer,"vectorizer.pkl")
joblib.dump(model,"model.pkl")

