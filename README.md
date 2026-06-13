# Sentiment Analysis Dashboard

An AI-powered web application that analyzes text and classifies its sentiment as **Positive**, **Negative**, or **Neutral**. Built using a complete machine learning pipeline — from raw data to a deployed, interactive dashboard.

🔗 **Live Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/KrishnaSR13/sentiment-dashboard)

---

## Overview

Companies like Amazon, Netflix, and Twitter receive millions of reviews, tweets, and comments daily — far more than any human team can read manually. Sentiment analysis automates this by identifying customer satisfaction, detecting complaints, and surfacing feedback trends at scale.

This project builds a sentiment classifier trained on real-world airline tweets and deploys it as an interactive dashboard where users can input any text and instantly see its predicted sentiment along with confidence scores.

---

## Dataset

| Detail | Info |
|---|---|
| Name | Twitter US Airline Sentiment |
| Source | Kaggle |
| Size | 14,640 tweets |
| Classes | Positive / Neutral / Negative |
| Type | Real customer tweets about US airlines |

---

## Pipeline

### 1. Data Loading
- Loaded the dataset using Pandas
- Retained only the `text` and `airline_sentiment` columns

### 2. Data Preprocessing
- Removed URLs, @mentions, hashtags, and special characters
- Converted text to lowercase
- Removed stopwords using NLTK

### 3. Feature Extraction
- TF-IDF Vectorizer with top 5,000 features
- Used bigrams (`ngram_range=(1,2)`) to capture two-word phrases like "not good" or "very happy"
- Converted cleaned text into numerical vectors

### 4. Model Training
- Split data 80% train / 20% test
- Trained a Logistic Regression classifier
- Used `class_weight='balanced'` to address class imbalance

### 5. Evaluation
- Achieved 76–79% accuracy on the test set
- Generated a confusion matrix and full classification report

### 6. Web Dashboard
- Built with Streamlit
- Displays a color-coded sentiment result (green / red / gray)
- Shows prediction confidence as a percentage
- Includes an interactive Plotly bar chart of confidence scores across all classes

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| NLP | NLTK, TF-IDF |
| ML Model | Scikit-learn |
| Dashboard | Streamlit |
| Charts | Plotly |
| Deployment | Hugging Face Spaces (Docker) |

---

## Project Structure

```
sentiment-dashboard/
├── app.py                Streamlit dashboard
├── train.py              Model training script
├── requirements.txt      Project dependencies
├── sentiment_model.pkl   Trained ML model
├── vectorizer.pkl         Saved TF-IDF vectorizer
└── cleaned_tweets.csv     Cleaned dataset
```

---

## Results

| Metric | Score |
|---|---|
| Overall Accuracy | 76–79% |
| Negative Detection | 82% precision |
| Positive Detection | 82% precision |
| Neutral Detection | 66% precision |

---

## Limitations & Future Improvements

The Twitter US Airline dataset is heavily skewed (~63% negative tweets), collected during a period of high customer complaints. This class imbalance causes two main issues:

- Short, neutral statements are sometimes misclassified as negative
- Negation patterns (e.g., "not good", "not pleasant") can be misclassified as positive, since TF-IDF weighs individual strong-sentiment words heavily even when negated

**Planned improvements:**
- Apply SMOTE oversampling to better balance the training classes
- Replace TF-IDF + Logistic Regression with a transformer-based model (e.g., BERT/DistilBERT), which understands word order and context far more effectively than bag-of-words approaches

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/krishna-s-r/sentiment-dashboard.git
cd sentiment-dashboard

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (optional — pre-trained model included)
python train.py

# 5. Run the dashboard
streamlit run app.py
```

---

## What This Project Demonstrates

| Concept | Description |
|---|---|
| NLP | Cleaning and preparing raw text for machine learning |
| TF-IDF | Converting text into numerical feature vectors |
| Logistic Regression | A classic, interpretable ML algorithm for classification |
| Class Imbalance | Identifying and addressing skewed training data |
| Model Evaluation | Using accuracy, precision, and confusion matrices |
| Deployment | Packaging an ML model into a live, interactive web app |

---

## Author

**Krishna S.R**
[GitHub](https://github.com/krishna-s-r) · [Hugging Face](https://huggingface.co/KrishnaSR13)
