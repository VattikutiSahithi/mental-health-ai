from flask import Flask, render_template, request
import joblib
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

app = Flask(__name__)

# -------------------
# PostgreSQL Database setup
# -------------------
DATABASE_URL = "postgresql+psycopg2://postgres:1234@localhost:5432/mental_health_ai"

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PatientRecord(Base):
    __tablename__ = "patient_records"
    id = Column(Integer, primary_key=True, index=True)
    gender = Column(String(20))
    age = Column(Integer)
    academic_pressure = Column(Integer)
    study_satisfaction = Column(Integer)
    sleep_duration = Column(Float)
    dietary_habits = Column(Integer)
    suicidal_thoughts = Column(Integer)
    work_study_hours = Column(Integer)
    financial_stress = Column(Integer)
    family_history = Column(Integer)
    depression_label = Column(Integer)
    statement_text = Column(Text)
    behavioral_risk = Column(String(10))
    sentiment = Column(String(20))
    crisis_level = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)


# Create table if not exists
Base.metadata.create_all(bind=engine)

# -------------------
# Load models
# -------------------
behavioral_model = joblib.load("models/behavioral_model.pkl")
scaler = joblib.load("models/scaler.pkl")
sentiment_model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/sentiment_vectorizer.pkl")

# -------------------
# Flask Routes
# -------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.form

    # Collect and clean inputs
    gender = int(data["gender"])
    age = int(data["age"])
    academic_pressure = int(data["academic_pressure"])
    study_satisfaction = int(data["study_satisfaction"])
    sleep_duration = float(data["sleep_duration"])
    dietary_habits = int(data["dietary_habits"])
    suicidal_thoughts = int(data["suicidal_thoughts"])
    work_study_hours = int(data["work_study_hours"])
    financial_stress = int(data["financial_stress"])
    family_history = int(data["family_history"])
    statement = data["statement"]

    # Predict behavioral risk
    features = np.array([[gender, age, academic_pressure, study_satisfaction,
                          sleep_duration, dietary_habits, suicidal_thoughts,
                          work_study_hours, financial_stress, family_history]])

    behavioral_pred = behavioral_model.predict(features)[0]
    behavioral_pred = int(behavioral_pred) if isinstance(behavioral_pred, np.generic) else behavioral_pred
    behavioral_risk = "Yes" if behavioral_pred == 1 else "No"

    # Predict sentiment
    text_features = vectorizer.transform([statement])
    sentiment_pred = sentiment_model.predict(text_features)[0]

    # Convert sentiment number → text
    sentiment_map = {
        1: "Very Negative",
        2: "Negative",
        3: "Neutral",
        4: "Positive",
        5: "Very Positive"
    }
    sentiment_text = sentiment_map.get(int(sentiment_pred), "Unknown")

    # Determine crisis level
    if behavioral_risk == "Yes" and sentiment_text in ["Very Negative", "Negative"]:
        crisis_level = "High"
    elif behavioral_risk == "Yes":
        crisis_level = "Medium"
    else:
        crisis_level = "Low"

    # Save record in database
    db = SessionLocal()
    record = PatientRecord(
        gender=str(gender),
        age=int(age),
        academic_pressure=int(academic_pressure),
        study_satisfaction=int(study_satisfaction),
        sleep_duration=float(sleep_duration),
        dietary_habits=int(dietary_habits),
        suicidal_thoughts=int(suicidal_thoughts),
        work_study_hours=int(work_study_hours),
        financial_stress=int(financial_stress),
        family_history=int(family_history),
        depression_label=int(behavioral_pred),
        statement_text=str(statement),
        behavioral_risk=str(behavioral_risk),
        sentiment=str(sentiment_text),  # ✅ Save text instead of number
        crisis_level=str(crisis_level)
    )

    db.add(record)
    db.commit()
    db.refresh(record)
    db.close()

    # Render results
    return render_template(
        "result.html",
        behavioral_risk=behavioral_risk,
        sentiment=sentiment_text,  # ✅ Show sentiment text
        crisis_level=crisis_level,
        record_id=record.id
    )


if __name__ == "__main__":
    app.run(debug=True)
