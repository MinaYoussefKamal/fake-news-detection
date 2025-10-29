import string
import joblib
import sv_ttk
from tkinter import *
from tkinter import ttk

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_news():
    text = text_box.get("1.0", END).strip()
    if not text:
        result_label.config(text="⚠️ Please enter some text!", foreground="#FFA500")
        confidence_label.config(text="")
        return

    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]

    try:
        confidence = model.decision_function(text_vec)[0]
        confidence = 1 / (1 + pow(2.71828, -confidence))
    except:
        confidence = 0.85

    if prediction == 1:
        result_label.config(text="📰 Real News", foreground="#00FA9A")
        confidence_label.config(
            text=f"Confidence: {round(confidence * 100, 2)}%", foreground="#00FA9A"
        )
    else:
        result_label.config(text="⚠️ Fake News", foreground="#F08080")
        confidence_label.config(
            text=f"Confidence: {round((1 - confidence) * 100, 2)}%", foreground="#F08080"
        )

def clear_text():
    text_box.delete("1.0", END)
    result_label.config(text="")
    confidence_label.config(text="")

window = Tk()
window.title("Fake News Detector 🧠")
window.geometry("700x550")
window.minsize(600, 450)

# Apply sv dark theme
sv_ttk.set_theme("dark")

# -------------------- GUI -------------------- #
frame = ttk.Frame(window, padding=30)
frame.place(relx=0.5, rely=0.5, anchor="center")

# Title
title_label = ttk.Label(
    frame, text="Fake News Detector 🧠",
    font=("Segoe UI Semibold", 22)
)
title_label.grid(column=0, row=0, columnspan=2, pady=(0, 20))

# Text Box
text_box = Text(
    frame, height=10, width=70,
    wrap="word", bg="#2A2A2A", fg="#FFF8E7",
    relief="flat", insertbackground="white", font=("Segoe UI", 11)
)
text_box.grid(column=0, row=1, columnspan=2, pady=(0, 25))

# Results
result_label = ttk.Label(
    frame, text="", font=("Segoe UI", 16, "bold")
)
result_label.grid(column=0, row=2, columnspan=2, pady=(10, 5))

confidence_label = ttk.Label(
    frame, text="", font=("Segoe UI", 12)
)
confidence_label.grid(column=0, row=3, columnspan=2, pady=(0, 20))

# Buttons
predict_button = ttk.Button(frame, text="Predict", command=predict_news)
predict_button.grid(column=0, row=4, padx=(0, 10), ipadx=10, ipady=6)

clear_button = ttk.Button(frame, text="Clear", command=clear_text)
clear_button.grid(column=1, row=4, padx=(10, 0), ipadx=10, ipady=6)

window.mainloop()
