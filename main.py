import pandas as pd
import re
import spacy
import joblib
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
stop_words = set(stopwords.words("english"))

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Combine title + text
data["content"] = (data["title"].fillna("") + " " + data["text"].fillna("")).astype(str)

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words]
    return " ".join(tokens)

print("🧹 Cleaning and lemmatizing texts (this may take a few minutes)...")
data["clean_text"] = data["content"].apply(preprocess)

x_train, x_test, y_train, y_test = train_test_split(
    data["clean_text"], data["label"], test_size=0.2, random_state=42
)

print(" Vectorizing...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

print("⚙️ Training SVM Model...")
model = LinearSVC()
model.fit(x_train_vec, y_train)

y_pred = model.predict(x_test_vec)
acc = round(accuracy_score(y_test, y_pred) * 100, 2)
f1 = round(f1_score(y_test, y_pred) * 100, 2)

print("\n📊 Model Evaluation:")
print(f"Accuracy: {acc}%")
print(f"F1 Score: {f1}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(model, "model.pkl")

print("\n✅ Model and vectorizer saved successfully.")
