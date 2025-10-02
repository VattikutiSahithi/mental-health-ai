# phase2_core_ai_models.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ==============================
# Depression Risk Prediction Model
# ==============================
def train_depression_model(csv_path=r"D:\Capstone\mental_health_ai\data\student_depression_cleaned.csv"):
    df = pd.read_csv(csv_path)

    # Ensure target is categorical (0/1)
    df['Depression'] = df['Depression'].astype(int)

    X = df.drop(columns=['Depression'])
    y = df['Depression']

    # Encode categorical features
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)   # ✅ FIXED
    acc = accuracy_score(y_test, preds)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/behavioral_model.pkl")

    return report, acc

# ==============================
# Sentiment / Statement Classifier
# ==============================
def train_sentiment_model(csv_path=r"D:\Capstone\mental_health_ai\Preprocessed_Data\sentiment_statements_ready.csv"):
    df = pd.read_csv(csv_path)

    # Use cleaned statements
    X = df["clean_statement"].astype(str)
    y = df["label"]

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_tfidf = vectorizer.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Train classifier
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save model + vectorizer
    joblib.dump(model, "models/sentiment_model.pkl")
    joblib.dump(vectorizer, "models/sentiment_vectorizer.pkl")

    return report, accuracy_score(y_test, y_pred)

# ==============================
# Main pipeline
# ==============================
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    results_summary = []

    # Train Depression Model
    dep_report, dep_acc = train_depression_model()
    dep_df = pd.DataFrame(dep_report).transpose()   # ✅ FIXED
    results_summary.append("=== Depression Risk Model ===\n")
    results_summary.append(f"Accuracy: {dep_acc:.4f}\n")
    results_summary.append(dep_df.to_string())
    results_summary.append("\n\n")

    # Train Sentiment Model
    sent_report, sent_acc = train_sentiment_model()
    sent_df = pd.DataFrame(sent_report).transpose()   # ✅ FIXED
    results_summary.append("=== Sentiment / Statement Model ===\n")
    results_summary.append(f"Accuracy: {sent_acc:.4f}\n")
    results_summary.append(sent_df.to_string())
    results_summary.append("\n")

    # Save evaluation results
    with open("evaluation_results.txt", "w", encoding="utf-8") as f:
        f.writelines(results_summary)

    print("✅ Training completed. Models and results saved.")
