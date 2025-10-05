import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ==============================
# Depression Risk Prediction Model (10 input features)
# ==============================
def train_depression_model(csv_path=r"D:\Capstone\mental_health_ai\data\student_depression_cleaned.csv"):
    df = pd.read_csv(csv_path)

    # Define relevant columns
    feature_cols = [
        'Gender',
        'Age',
        'Academic Pressure',
        'Study Satisfaction',
        'Sleep Duration',
        'Dietary Habits',
        'Have you ever had suicidal thoughts ?',
        'Work/Study Hours',
        'Financial Stress',
        'Family History of Mental Illness'
    ]
    target_col = 'Depression'

    # Keep required columns only
    df = df[feature_cols + [target_col]].copy()

    # Clean and encode categorical values
    for col in feature_cols:
        df[col] = df[col].astype(str).str.strip().str.replace("'", "", regex=False)
        df[col] = df[col].astype('category').cat.codes

    X = df[feature_cols]
    y = df[target_col].astype(int)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate
    report = classification_report(y_test, y_pred, output_dict=True)
    acc = accuracy_score(y_test, y_pred)

    # Save model + scaler + features
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/behavioral_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_cols, "models/behavioral_features.pkl")

    return report, acc

# ==============================
# Sentiment / Statement Classifier
# ==============================
def train_sentiment_model(csv_path=r"D:\Capstone\mental_health_ai\Preprocessed_Data\sentiment_statements_ready.csv"):
    df = pd.read_csv(csv_path)

    X = df["clean_statement"].astype(str)
    y = df["label"].astype(str)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    acc = accuracy_score(y_test, y_pred)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/sentiment_model.pkl")
    joblib.dump(vectorizer, "models/sentiment_vectorizer.pkl")

    return report, acc

# ==============================
# Main pipeline
# ==============================
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    results_summary = []

    # Train Depression Model
    dep_report, dep_acc = train_depression_model()
    dep_df = pd.DataFrame(dep_report).transpose()
    results_summary.append("=== Depression Risk Model ===\n")
    results_summary.append(f"Accuracy: {dep_acc:.4f}\n")
    results_summary.append(dep_df.to_string())
    results_summary.append("\n\n")

    # Train Sentiment Model
    sent_report, sent_acc = train_sentiment_model()
    sent_df = pd.DataFrame(sent_report).transpose()
    results_summary.append("=== Sentiment / Statement Model ===\n")
    results_summary.append(f"Accuracy: {sent_acc:.4f}\n")
    results_summary.append(sent_df.to_string())
    results_summary.append("\n")

    with open("evaluation_results.txt", "w", encoding="utf-8") as f:
        f.writelines(results_summary)

    print("✅ Training completed successfully! Models and results saved in the 'models' folder.")
