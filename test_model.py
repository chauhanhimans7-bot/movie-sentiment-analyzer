import re
import html
import json
import numpy as np

from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences


# --------------------------------
# 1. Define file paths
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "sentiment_model.keras"
TOKENIZER_PATH = BASE_DIR / "tokenizer.json"

MAXLEN = 600


# --------------------------------
# 2. Load trained model
# --------------------------------

model = load_model(MODEL_PATH, compile=False)

print("LSTM model loaded successfully!")


# --------------------------------
# 3. Load tokenizer
# --------------------------------

with open(TOKENIZER_PATH, "r", encoding="utf-8") as file:
    tokenizer = tokenizer_from_json(file.read())

print("Tokenizer loaded successfully!")


# --------------------------------
# 4. Text cleaning
# --------------------------------

def clean_text(text):

    text = text.lower()

    text = html.unescape(text)

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'http\S+|www\.\S+', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


# --------------------------------
# 5. Prediction function
# --------------------------------

def predict_sentiment(review):

    cleaned_review = clean_text(review)

    sequence = tokenizer.texts_to_sequences([cleaned_review])

    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAXLEN,
        padding="post",
        truncating="post"
    )

    probability = float(
        model.predict(padded_sequence, verbose=0)[0][0]
    )

    sentiment = "Positive" if probability >= 0.5 else "Negative"

    confidence = (
        probability if probability >= 0.5
        else 1 - probability
    ) * 100

    print("\nReview:", review)
    print("Sentiment:", sentiment)
    print("Confidence:", round(confidence, 2), "%")


# --------------------------------
# 6. Test prediction
# --------------------------------

review = "This movie was absolutely amazing. I loved every moment of it!"

predict_sentiment(review)
