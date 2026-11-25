from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(190), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    cases = db.relationship("Case", backref="user", cascade="all, delete-orphan")

class Case(db.Model):
    __tablename__ = "cases"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    patient_label = db.Column(db.String(120), nullable=False)
    notes = db.Column(db.Text)
    image_filename = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(40), default="uploaded")
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    analyses = db.relationship("Analysis", backref="case", cascade="all, delete-orphan")

class Analysis(db.Model):
    __tablename__ = "analyses"
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey("cases.id"), nullable=False)
    status = db.Column(db.String(40), nullable=False)
    mask_filename = db.Column(db.String(255))
    stenosis_percent = db.Column(db.Float)
    message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
