# app.py

'''import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import joblib
import os

# ---------- Step 1: Load or Train Model ----------
MODEL_FILE = "emotion_model.pkl"
VECT_FILE = "vectorizer.pkl"

SAMPLE_TRAIN_ROWS = 2000
SAMPLE_VAL_ROWS = 500
SAMPLE_TEST_ROWS = 500

if os.path.exists(MODEL_FILE) and os.path.exists(VECT_FILE):
    # Load trained model
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECT_FILE)
else:
    # Load CSVs
    train_df = pd.read_csv("training.csv")
    val_df = pd.read_csv("validation.csv")
    test_df = pd.read_csv("test.csv")

    # ---------- Take small subset for fast training ----------
    train_df = train_df.sample(min(SAMPLE_TRAIN_ROWS, len(train_df)), random_state=42)
    val_df = val_df.sample(min(SAMPLE_VAL_ROWS, len(val_df)), random_state=42)
    test_df = test_df.sample(min(SAMPLE_TEST_ROWS, len(test_df)), random_state=42)

    # ---------- Detect text and label columns ----------
    text_col = None
    label_col = None
    for col in train_df.columns:
        if train_df[col].dtype == object and train_df[col].nunique() > 5:
            text_col = col
        else:
            label_col = col

    if text_col is None or label_col is None:
        st.error("Could not detect text/label columns in CSV!")
        st.stop()

    # Preprocessing
    def preprocess_text(df):
        df[text_col] = df[text_col].str.lower()
        df[text_col] = df[text_col].str.replace(r'[^a-z0-9\s]', '', regex=True)
        return df

    train_df = preprocess_text(train_df)
    val_df = preprocess_text(val_df)
    test_df = preprocess_text(test_df)

    # Feature extraction
    vectorizer = CountVectorizer(max_features=500)
    X_train = vectorizer.fit_transform(train_df[text_col])
    y_train = train_df[label_col]

    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Save model for future use
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECT_FILE)

# ---------- Step 2: Streamlit UI ----------
st.title("💬 Emotion Detection App")
st.write("Enter a message to detect its emotion:")

user_input = st.text_area("Your message here:")

if st.button("Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter a message!")
    else:
        # Preprocess input
        text = user_input.lower()
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        vect = vectorizer.transform([text])
        prediction = model.predict(vect)[0]
        st.success(f"Predicted Emotion: **{prediction}**")'''
# app.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import joblib
import os

# ---------- Step 1: Load or Train Model ----------
MODEL_FILE = "emotion_model.pkl"
VECT_FILE = "vectorizer.pkl"

SAMPLE_TRAIN_ROWS = 2000
SAMPLE_VAL_ROWS = 500
SAMPLE_TEST_ROWS = 500

# Mapping numeric labels to emotion names
emotion_mapping = {
    0: "anger",
    1: "happy",
    2: "sad",
    3: "fear",
    4: "surprise",
    5: "disgust"
}

if os.path.exists(MODEL_FILE) and os.path.exists(VECT_FILE):
    # Load trained model
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECT_FILE)
else:
    # Load CSVs
    train_df = pd.read_csv("training.csv")
    val_df = pd.read_csv("validation.csv")
    test_df = pd.read_csv("test.csv")

    # ---------- Take small subset for fast training ----------
    train_df = train_df.sample(min(SAMPLE_TRAIN_ROWS, len(train_df)), random_state=42)
    val_df = val_df.sample(min(SAMPLE_VAL_ROWS, len(val_df)), random_state=42)
    test_df = test_df.sample(min(SAMPLE_TEST_ROWS, len(test_df)), random_state=42)

    # ---------- Detect text and label columns ----------
    text_col = None
    label_col = None
    for col in train_df.columns:
        if train_df[col].dtype == object and train_df[col].nunique() > 5:
            text_col = col
        else:
            label_col = col

    if text_col is None or label_col is None:
        st.error("Could not detect text/label columns in CSV!")
        st.stop()

    # Preprocessing
    def preprocess_text(df):
        df[text_col] = df[text_col].str.lower()
        df[text_col] = df[text_col].str.replace(r'[^a-z0-9\s]', '', regex=True)
        return df

    train_df = preprocess_text(train_df)
    val_df = preprocess_text(val_df)
    test_df = preprocess_text(test_df)

    # Feature extraction
    vectorizer = CountVectorizer(max_features=500)
    X_train = vectorizer.fit_transform(train_df[text_col])
    y_train = train_df[label_col].astype(int)  # ensure numeric labels

    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Save model for future use
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECT_FILE)

# ---------- Step 2: Streamlit UI ----------
st.title("💬 Emotion Detection App")
st.write("Enter a message to detect its emotion:")

user_input = st.text_area("Your message here:")

if st.button("Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter a message!")
    else:
        # Preprocess input
        text = user_input.lower()
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        vect = vectorizer.transform([text])
        predicted_label = model.predict(vect)[0]
        prediction_name = emotion_mapping.get(predicted_label, "Unknown")
        st.success(f"Predicted Emotion: **{prediction_name}**")
