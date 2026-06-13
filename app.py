import streamlit as st
import pickle
import re
import plotly.graph_objects as go
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)

# Load model and vectorizer
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = round(max(proba) * 100, 2)
    return prediction, confidence, proba

# Page config
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🎭",
    layout="centered"
)

# Title
st.title("Sentiment Analysis Dashboard")
st.markdown("Analyze the sentiment of any text — Positive, Neutral, or Negative.")
st.markdown("---")

# Input
text_input = st.text_area("Enter your text below:", height=150,
                           placeholder="Type any tweet, review or comment here...")

if st.button("Analyze Sentiment"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction, confidence, proba = predict_sentiment(text_input)

        # Color based on sentiment
        color_map = {
            'positive': '#2ECC71',
            'negative': '#E74C3C',
            'neutral': '#95A5A6'
        }
        label_map = {
            'positive': 'POSITIVE',
            'negative': 'NEGATIVE',
            'neutral': 'NEUTRAL'
        }

        color = color_map[prediction]
        label = label_map[prediction]

        st.markdown("---")

        # Result box
        st.markdown(
            f"<div style='background-color:{color};padding:20px;border-radius:10px;text-align:center'>"
            f"<h2 style='color:white'>{label}</h2>"
            f"<h4 style='color:white'>Confidence: {confidence}%</h4>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown("---")

        # Confidence chart
        st.subheader("Confidence Score Breakdown")
        classes = model.classes_
        fig = go.Figure(go.Bar(
            x=[p * 100 for p in proba],
            y=classes,
            orientation='h',
            marker_color=['#E74C3C', '#95A5A6', '#2ECC71']
        ))
        fig.update_layout(
            xaxis_title="Confidence (%)",
            yaxis_title="Sentiment",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
