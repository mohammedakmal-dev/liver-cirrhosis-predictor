from flask import Flask, render_template, request, session, make_response, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from weasyprint import HTML
from datetime import datetime
import joblib
import numpy as np
from models import db, User, Prediction

app = Flask(__name__)
app.secret_key = "super-secret-key"  # Replace in production

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"

# Load model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("normalizer.pkl")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            form_data = request.form.to_dict()
            data = [float(x) for x in form_data.values()]
            data = np.array(data).reshape(1, -1)
            scaled = scaler.transform(data)
            result = model.predict(scaled)[0]
            prediction = "Cirrhosis Detected (Class 1)" if result == 1 else "No Cirrhosis Detected (Class 2)"

            session['inputs'] = form_data
            session['result'] = prediction

            # If user is logged in, save prediction
            if current_user.is_authenticated:
                new_pred = Prediction(
                    input_data=str(form_data),
                    result=prediction,
                    timestamp=datetime.now(),
                    user_id=current_user.id
                )
                db.session.add(new_pred)
                db.session.commit()
        except Exception as e:
            prediction = "Something went wrong. Please check your inputs."
            print("Error:", e)

    return render_template("index.html", result=prediction)

@app.route("/download_report")
@login_required
def download_report():
    if 'inputs' in session and 'result' in session:
        html = render_template(
            "report.html",
            inputs=session['inputs'],
            result=session['result'],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M")
        )
        pdf = HTML(string=html).write_pdf()
        response = make_response(pdf)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=prediction_report.pdf'
        return response
    return redirect(url_for("index"))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if User.query.filter_by(username=username).first():
            flash("Username already exists.")
        else:
            hashed = generate_password_hash(password)
            new_user = User(username=username, password=hashed)
            db.session.add(new_user)
            db.session.commit()
            flash("Signup successful! Please login.")
            return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("index"))
        else:
            flash("Invalid credentials.")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.")
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    history = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    return render_template("dashboard.html", history=history)
