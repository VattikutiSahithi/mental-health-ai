import os

DATABASE_URL = "postgresql+psycopg2://postgres:1234@localhost:5432/mental_health_ai"

class Config:
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = "supersecretkey"
