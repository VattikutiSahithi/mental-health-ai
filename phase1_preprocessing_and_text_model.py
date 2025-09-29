# phase1_preprocessing_and_text_model.py

import os
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ---------------------------
# USER CONFIG
# ---------------------------
BASE_DIR = r"D:\Capstone\mental_health_ai"
STRUCTURED_CSV = os.path.join(BASE_DIR,"Preprocessed_Data", "student_depression_ready.csv")
TEXT_CSV = os.path.join(BASE_DIR,"Preprocessed_Data", "sentiment_statements_ready.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Preprocessed_Data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------
# 1. Structured Dataset Preprocessing
# ---------------------------
print("📌 Loading structured dataset...")
df_structured = pd.read_csv(STRUCTURED_CSV)

# Drop duplicates
df_structured.drop_duplicates(inplace=True)

# Encode categorical columns
categorical_cols = df_structured.select_dtypes(include=['object']).columns
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_structured[col] = le.fit_transform(df_structured[col].astype(str))
    le_dict[col] = le
    joblib.dump(le, os.path.join(MODEL_DIR, f"le_{col}.pkl"))

# Scale numeric columns
numeric_cols = df_structured.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
df_structured[numeric_cols] = scaler.fit_transform(df_structured[numeric_cols])
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# Save preprocessed structured dataset
structured_out = os.path.join(OUTPUT_DIR, "student_depression_ready.csv")
df_structured.to_csv(structured_out, index=False)
print(f"✅ Structured dataset saved: {structured_out}")

# ---------------------------
# 2. Text Dataset Preprocessing
# ---------------------------
print("\n📌 Loading sentiment text dataset...")
df_text = pd.read_csv(TEXT_CSV)

# Keep only statement + status
df_text = df_text[['statement', 'status']]

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text

df_text['clean_statement'] = df_text['statement'].apply(clean_text)

# Encode labels
label_encoder = LabelEncoder()
df_text['label'] = label_encoder.fit_transform(df_text['status'])
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df_text['clean_statement'],
    df_text['label'],
    test_size=0.2,
    random_state=42,
    stratify=df_text['label']
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

# ---------------------------
# 3. Baseline Text Classification Model
# ---------------------------
print("\n📌 Training baseline Logistic Regression model for text classification...")
model = LogisticRegression(max_iter=500)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print("\n📊 Classification Report:")
print(report)
joblib.dump(model, os.path.join(MODEL_DIR, "sentiment_logreg_model.pkl"))
print("✅ Baseline text classification model saved.")

# ---------------------------
# 4. Save Preprocessed Sentiment Dataset
# ---------------------------
sentiment_out = os.path.join(OUTPUT_DIR, "sentiment_statements_ready.csv")
df_text.to_csv(sentiment_out, index=False)
print(f"✅ Sentiment dataset saved: {sentiment_out}")

# ---------------------------
# 5. Save Evaluation Summary
# ---------------------------
results_file = os.path.join(OUTPUT_DIR, "results.txt")
with open(results_file, "w", encoding="utf-8") as f:
    f.write("📊 Sentiment Model Evaluation Summary\n")
    f.write("====================================\n\n")
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
print(f"✅ Evaluation summary saved: {results_file}")

print("\n🚀 Phase 1 preprocessing + baseline model training completed!")
