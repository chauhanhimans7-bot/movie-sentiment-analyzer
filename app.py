
import streamlit as st
import re
import html
import time 
import numpy as np
from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "sentiment_model.keras"
TOKENIZER_PATH = BASE_DIR / "tokenizer.json"

MAXLEN = 600


# ==========================================
# 2. STREAMLIT PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)


# ==========================================
# 3. LOAD MODEL AND TOKENIZER
# ==========================================

@st.cache_resource
def load_artifacts():

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    with open(TOKENIZER_PATH, "r", encoding="utf-8") as file:
        tokenizer = tokenizer_from_json(file.read())

    # Warm up the model once
    dummy_input = np.zeros((1, MAXLEN), dtype=np.int32)
    model(dummy_input, training=False)

    return model, tokenizer

# ==========================================
# 4. TEXT PREPROCESSING
# ==========================================

def clean_text(text):

    text = text.lower()

    text = html.unescape(text)

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'http\S+|www\.\S+', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ==========================================
# 5. SENTIMENT PREDICTION
# ==========================================


def predict_sentiment(review, model, tokenizer):

    cleaned_review = clean_text(review)

    sequence = tokenizer.texts_to_sequences([cleaned_review])

    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAXLEN,
        padding="post",
        truncating="post"
    )

    # Direct inference instead of model.predict()
    prediction = model(
        padded_sequence,
        training=False
    )

    positive_probability = float(prediction.numpy()[0][0])

    sentiment = "Positive" if positive_probability >= 0.5 else "Negative"

    score = (
        positive_probability
        if sentiment == "Positive"
        else 1 - positive_probability
    )

    return sentiment, score, positive_probability


# Load model and tokenizer once
try:
    with st.spinner("Loading sentiment model..."):
        model, tokenizer = load_artifacts()

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# ==========================================
# 6. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #172554,
        #111827
    );
    color: #f8fafc;
}

/* Main content */
.block-container {
    max-width: 850px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

/* Header */
.hero {
    text-align: center;
    padding: 20px 10px 30px 10px;
    animation: fadeIn 1s ease-in-out;
}

.hero h1 {
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -1px;
}

.hero p {
    color: #cbd5e1;
    font-size: 17px;
}

/* Text area */
.stTextArea textarea {
    background-color: #1e293b;
    color: white;
    border: 1px solid #475569;
    border-radius: 12px;
}

/* Analyze button */
.stButton button {
    width: 100%;
    border: none;
    border-radius: 10px;
    padding: 12px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    font-weight: 700;
    transition: all 0.3s ease;
}

.stButton button:hover {
    transform: translateY(-3px);
    box-shadow: 0 5px 20px #6366f180;
    color: white;
}

/* Result card */
.result-card {
    padding: 30px;
    border-radius: 18px;
    background: rgba(30, 41, 59, 0.85);
    text-align: center;
    border: 1px solid #475569;
    animation: fadeIn 0.7s ease-in-out;
}

/* Animations */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(15px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    margin-top: 50px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 7. APPLICATION HEADER
# ==========================================

st.markdown("""
<div class="hero">

    <h1>🎬 Movie Sentiment Analyzer</h1>

    <p>
        Discover the emotions behind movie reviews
        using Deep Learning and LSTM.
    </p>

</div>
""", unsafe_allow_html=True)


# ==========================================
# 8. REVIEW INPUT
# ==========================================

st.subheader("📝 Enter Your Movie Review")

with st.form("sentiment_form"):

    review = st.text_area(
        "Write your review below:",
        placeholder="Example: This movie was absolutely amazing. I loved every moment of it!",
        height=180
    )

    analyze_button = st.form_submit_button(
        "✨ Analyze Sentiment"
    )



# ==========================================
# 9. ANALYZE REVIEW
# ==========================================

if analyze_button:

    if not review.strip():

        st.warning("Please enter a movie review first.")

    else:

        try:
            with st.spinner("Analyzing your review..."):

                start_time = time.perf_counter()

                sentiment, score, positive_probability = (
                    predict_sentiment(review, model, tokenizer)
                )

                prediction_time = time.perf_counter() - start_time

                st.session_state["result"] = {
                    "sentiment": sentiment,
                    "score": score,
                    "positive_probability": positive_probability,
                    "prediction_time": prediction_time
                }

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ==========================================
# 10. DISPLAY RESULT
# ==========================================

if "result" in st.session_state:

    result = st.session_state["result"]

    sentiment = result["sentiment"]
    score = result["score"]
    positive_probability = result["positive_probability"]
    prediction_time = result["prediction_time"]

    st.divider()

    st.subheader("Analysis Results")

    # Sentiment result
    if sentiment == "Positive":

        st.success("😊 Positive Sentiment Detected")

    else:

        st.error("☹️ Negative Sentiment Detected")

    # Display metrics
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Predicted Sentiment",
            value=sentiment
        )

    with col2:
        st.metric(
            label="Prediction Score",
            value=f"{score * 100:.2f}%"
        )

    # Visual score
    st.write("**Prediction Score**")

    st.progress(float(score))

    # Additional information
    with st.expander("View prediction details"):

        st.write(
            f"Positive probability: {positive_probability * 100:.2f}%"
        )

        st.write(
            f"Negative probability: {(1 - positive_probability) * 100:.2f}%"
        )

        st.write(
            f"Prediction time: {prediction_time:.3f} seconds"
        )

    # Clear result
    if st.button("🔄 Analyze Another Review"):

        del st.session_state["result"]
        st.rerun()


# ==========================================
# 11. FOOTER
# ==========================================

st.markdown("""
<div class="footer">

    Built with Python | TensorFlow | LSTM | Streamlit

    <br><br>

    Movie Review Sentiment Analysis Project

</div>
""", unsafe_allow_html=True)
