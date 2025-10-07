from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, func
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timedelta

app = Flask(__name__)

# -------------------
# PostgreSQL Database setup
# -------------------
DATABASE_URL = "postgresql+psycopg2://postgres:1234@localhost:5432/mental_health_ai"

engine = create_engine(DATABASE_URL, echo=False)
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

    # -----------------------------
    # FIXED SECTION: Apply same scaling used during training
    # -----------------------------
    raw_features = np.array([[gender, age, academic_pressure, study_satisfaction,
                              sleep_duration, dietary_habits, suicidal_thoughts,
                              work_study_hours, financial_stress, family_history]])

    # Apply StandardScaler transformation
    scaled_features = scaler.transform(raw_features)

    # Predict behavioral risk using scaled inputs
    behavioral_pred = behavioral_model.predict(scaled_features)[0]
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
        sentiment=str(sentiment_text),
        crisis_level=str(crisis_level)
    )

    db.add(record)
    db.commit()
    db.refresh(record)
    db.close()

    return render_template(
        "result.html",
        behavioral_risk=behavioral_risk,
        sentiment=sentiment_text,
        crisis_level=crisis_level,
        record_id=record.id
    )


# -------------------
# Admin Dashboard Routes
# -------------------
@app.route("/admin")
def admin_dashboard():
    """Main admin dashboard with overview statistics"""
    db = SessionLocal()

    try:
        # Get total patients
        total_patients = db.query(PatientRecord).count()

        # Get crisis level distribution
        crisis_stats = db.query(
            PatientRecord.crisis_level,
            func.count(PatientRecord.id).label('count')
        ).group_by(PatientRecord.crisis_level).all()

        # Convert Row objects to tuples for template compatibility
        crisis_stats = [(row.crisis_level, row.count) for row in crisis_stats]

        # Get recent patients (last 7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_patients = db.query(PatientRecord).filter(
            PatientRecord.created_at >= week_ago
        ).count()

        # Get high-risk patients
        high_risk_patients = db.query(PatientRecord).filter(
            PatientRecord.crisis_level == "High"
        ).count()

        # Get average age
        avg_age = db.query(func.avg(PatientRecord.age)).scalar() or 0

        # Get gender distribution
        gender_stats = db.query(
            PatientRecord.gender,
            func.count(PatientRecord.id).label('count')
        ).group_by(PatientRecord.gender).all()

        # Convert Row objects to tuples for template compatibility
        gender_stats = [(row.gender, row.count) for row in gender_stats]

        # Get sentiment distribution
        sentiment_stats = db.query(
            PatientRecord.sentiment,
            func.count(PatientRecord.id).label('count')
        ).group_by(PatientRecord.sentiment).all()

        # Convert Row objects to tuples for template compatibility
        sentiment_stats = [(row.sentiment, row.count) for row in sentiment_stats]

        # Get recent assessments (last 10)
        recent_assessments = db.query(PatientRecord).order_by(
            PatientRecord.created_at.desc()
        ).limit(10).all()

        return render_template("admin/dashboard.html",
                               total_patients=total_patients,
                               crisis_stats=crisis_stats,
                               recent_patients=recent_patients,
                               high_risk_patients=high_risk_patients,
                               avg_age=round(avg_age, 1),
                               gender_stats=gender_stats,
                               sentiment_stats=sentiment_stats,
                               recent_assessments=recent_assessments)
    except Exception as e:
        print(f"Database error: {e}")
        return f"Database error: {e}", 500
    finally:
        db.close()


@app.route("/admin/patients")
def admin_patients():
    """Patient management page"""
    db = SessionLocal()

    try:
        # Get filter parameters
        crisis_filter = request.args.get('crisis_level', '')
        search_term = request.args.get('search', '')
        page = int(request.args.get('page', 1))
        per_page = 20

        # Build query
        query = db.query(PatientRecord)

        if crisis_filter:
            query = query.filter(PatientRecord.crisis_level == crisis_filter)

        if search_term:
            query = query.filter(
                PatientRecord.statement_text.ilike(f'%{search_term}%')
            )

        # Get total count for pagination
        total_count = query.count()

        # Get paginated results
        patients = query.order_by(PatientRecord.created_at.desc()).offset(
            (page - 1) * per_page
        ).limit(per_page).all()

        # Calculate pagination info
        total_pages = (total_count + per_page - 1) // per_page

        return render_template("admin/patients.html",
                               patients=patients,
                               crisis_filter=crisis_filter,
                               search_term=search_term,
                               page=page,
                               total_pages=total_pages,
                               total_count=total_count)
    finally:
        db.close()


@app.route("/admin/patient/<int:patient_id>")
def admin_patient_detail(patient_id):
    """Detailed patient report"""
    db = SessionLocal()

    try:
        patient = db.query(PatientRecord).filter(
            PatientRecord.id == patient_id
        ).first()

        if not patient:
            return "Patient not found", 404

        # Generate risk analysis
        risk_factors = []
        if patient.suicidal_thoughts == 1:
            risk_factors.append("Suicidal thoughts reported")
        if patient.family_history == 1:
            risk_factors.append("Family history of mental illness")
        if patient.academic_pressure >= 8:
            risk_factors.append("High academic pressure")
        if patient.financial_stress >= 8:
            risk_factors.append("High financial stress")
        if patient.sleep_duration < 6:
            risk_factors.append("Insufficient sleep")
        if patient.dietary_habits <= 3:
            risk_factors.append("Poor dietary habits")

        # Generate recommendations
        recommendations = []
        if patient.crisis_level == "High":
            recommendations.extend([
                "Immediate professional intervention required",
                "Consider crisis intervention services",
                "Schedule urgent follow-up appointment",
                "Implement safety planning"
            ])
        elif patient.crisis_level == "Medium":
            recommendations.extend([
                "Schedule follow-up within 1-2 weeks",
                "Consider therapy or counseling",
                "Monitor for symptom changes",
                "Provide coping strategies"
            ])
        else:
            recommendations.extend([
                "Continue current care plan",
                "Regular check-ins recommended",
                "Maintain healthy lifestyle habits",
                "Monitor for any changes"
            ])

        return render_template("admin/patient_detail.html",
                               patient=patient,
                               risk_factors=risk_factors,
                               recommendations=recommendations)
    finally:
        db.close()


@app.route("/admin/analytics")
def admin_analytics():
    """Analytics and trends page"""
    db = SessionLocal()

    try:
        # Get data for charts
        # Crisis level trends over time (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        daily_crisis = db.query(
            func.date(PatientRecord.created_at).label('date'),
            PatientRecord.crisis_level,
            func.count(PatientRecord.id).label('count')
        ).filter(
            PatientRecord.created_at >= thirty_days_ago
        ).group_by(
            func.date(PatientRecord.created_at),
            PatientRecord.crisis_level
        ).all()

        # Convert Row objects to tuples for template compatibility
        daily_crisis = [(row.date, row.crisis_level, row.count) for row in daily_crisis]

        # Age distribution - using simpler approach
        age_groups = []
        age_ranges = [
            ('Under 18', PatientRecord.age < 18),
            ('18-25', PatientRecord.age.between(18, 25)),
            ('26-35', PatientRecord.age.between(26, 35)),
            ('36-50', PatientRecord.age.between(36, 50)),
            ('Over 50', PatientRecord.age > 50)
        ]

        for age_label, condition in age_ranges:
            count = db.query(PatientRecord).filter(condition).count()
            if count > 0:
                age_groups.append((age_label, count))

        # Risk factors analysis
        risk_analysis = {
            'suicidal_thoughts': db.query(PatientRecord).filter(
                PatientRecord.suicidal_thoughts == 1
            ).count(),
            'family_history': db.query(PatientRecord).filter(
                PatientRecord.family_history == 1
            ).count(),
            'high_academic_pressure': db.query(PatientRecord).filter(
                PatientRecord.academic_pressure >= 8
            ).count(),
            'high_financial_stress': db.query(PatientRecord).filter(
                PatientRecord.financial_stress >= 8
            ).count(),
            'poor_sleep': db.query(PatientRecord).filter(
                PatientRecord.sleep_duration < 6
            ).count()
        }

        # Sentiment analysis trends
        sentiment_trends = db.query(
            PatientRecord.sentiment,
            func.count(PatientRecord.id).label('count')
        ).group_by(PatientRecord.sentiment).all()

        # Convert Row objects to tuples for template compatibility
        sentiment_trends = [(row.sentiment, row.count) for row in sentiment_trends]

        return render_template("admin/analytics.html",
                               daily_crisis=daily_crisis,
                               age_groups=age_groups,
                               risk_analysis=risk_analysis,
                               sentiment_trends=sentiment_trends)
    finally:
        db.close()


@app.route("/admin/api/stats")
def admin_api_stats():
    """API endpoint for dashboard statistics"""
    db = SessionLocal()

    try:
        # Get real-time statistics
        stats = {
            'total_patients': db.query(PatientRecord).count(),
            'high_risk_count': db.query(PatientRecord).filter(
                PatientRecord.crisis_level == "High"
            ).count(),
            'medium_risk_count': db.query(PatientRecord).filter(
                PatientRecord.crisis_level == "Medium"
            ).count(),
            'low_risk_count': db.query(PatientRecord).filter(
                PatientRecord.crisis_level == "Low"
            ).count(),
            'today_assessments': db.query(PatientRecord).filter(
                func.date(PatientRecord.created_at) == datetime.utcnow().date()
            ).count()
        }

        return jsonify(stats)
    finally:
        db.close()


if __name__ == "__main__":
    app.run(debug=True)
