# 📰 Fake News Detector — NLP Project

This is my **third NLP project**, built as part of my continued learning in Natural Language Processing and Machine Learning.
It’s a **Fake News Detector** that classifies news articles as *Real* or *Fake* based on their text content — featuring a clean **Tkinter GUI with the Sun Valley (sv-ttk) dark theme** for modern styling.

---

## 🚀 Project Overview

This project combines **text preprocessing, vectorization, and model training** to identify fake news.
It uses **SpaCy** for lemmatization, **TF-IDF Vectorization** for feature extraction, and a **Linear Support Vector Classifier (LinearSVC)** for classification.
The GUI makes it easy for users to input text and instantly get predictions along with confidence scores.

---

## 🧠 Key Features

* **Trained on the Kaggle Fake and True News dataset**
* **Lemmatization** with SpaCy’s `en_core_web_sm` model
* **TF-IDF Vectorization** to convert text into numerical features
* **LinearSVC Model** for efficient binary classification
* **Beautiful GUI** built using Tkinter + Sun Valley (`sv-ttk`) theme
* Displays **prediction confidence** and user-friendly results

---

## 🗂️ Files

* `main.py` → Handles **data loading, preprocessing, training**, and **model saving**
* `predict.py` → Provides a **graphical interface** for real-time predictions

---

## ⚙️ Requirements

Make sure you have these installed before running the project:

```bash
pip install pandas scikit-learn nltk spacy joblib sv-ttk
python -m spacy download en_core_web_sm
```

---

## 📥 Dataset Download

You must **manually download** the dataset before running the training script.
The project uses the **Kaggle “Fake and True News” dataset**:

🔗 [Fake and True News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

After downloading:

* Place the files **`Fake.csv`** and **`True.csv`** in the same directory as your scripts.

---

## ▶️ How to Run

### 1️⃣ Train the Model

Run `main.py` to preprocess the data, train the model, and save the vectorizer + model:

```bash
python main.py
```

### 2️⃣ Launch the GUI

Run `predict.py` to open the graphical interface and test your own text inputs:

```bash
python predict.py
```

---

## 🖼️ Example

**Input:**

> "Government confirms new policy on renewable energy."

**Output:**
✅ Real News (Confidence: 94.7%)

---

## 💡 About This Project

This is my **third NLP project**, part of my journey in the **Elevvo Pathways Internship** and my exploration of machine learning, NLP, and GUI development with Python.
I’m continuously learning how to bridge **AI models** and **user-friendly interfaces** — and this project reflects that growth.

---

### 📌 Author

**Mina Youssef Kamal**
🔗 [GitHub: MinaYoussefKamal](https://github.com/MinaYoussefKamal)
