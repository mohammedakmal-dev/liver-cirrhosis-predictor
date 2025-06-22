from flask import Flask, render_template, request, session, make_response, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from weasyprint import HTML
from datetime import datetime
from collections import Counter
import joblib
import numpy as np
import csv
import io
import os
import shap
from models import db, User, Prediction

app = Flask(__name__)
app.secret_key = "super-secret-key"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"

model = joblib.load("rf_model.pkl")
scaler = joblib.load("normalizer.pkl")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.before_request
def initialize_database():
    if not os.path.exists("users.db"):
        with app.app_context():
            db.create_all()
            print("✅ Database created on first request")

@app.route("/health")
def health():
    return "OK"

from datetime import datetime

@app.route("/")
def home():
    return render_template("home.html", datetime=datetime)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    explanation = None
    if request.method == "POST":
        try:
            form_data = request.form.to_dict()
            data = [float(x) for x in form_data.values()]
            data = np.array(data).reshape(1, -1)
            scaled = scaler.transform(data)
            result = model.predict(scaled)[0]
            prediction = "Cirrhosis Detected (Class 1)" if result == 1 else "No Cirrhosis Detected (Class 2)"

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(scaled)
            feature_names = list(form_data.keys())
            shap_explanation = dict(zip(feature_names, shap_values[1][0]))
            sorted_explanation = dict(sorted(shap_explanation.items(), key=lambda x: abs(x[1]), reverse=True))

            shap_plot = {
                "features": list(sorted_explanation.keys())[:10],
                "values": [round(float(v), 4) for v in list(sorted_explanation.values())[:10]]
            }

            session['inputs'] = form_data
            session['result'] = prediction
            session['explanation'] = sorted_explanation
            session['shap_plot'] = shap_plot

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
            print("🚨 Prediction Error:", e)

    return render_template("index.html", result=prediction, explanation=session.get("explanation"), shap_plot=session.get("shap_plot"), datetime=datetime)

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
    return redirect(url_for("predict"))


from datetime import datetime
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        try:
            if User.query.filter_by(username=username).first():
                flash("Username already exists.")
            else:
                hashed = generate_password_hash(password)
                new_user = User(username=username, password=hashed)
                db.session.add(new_user)
                db.session.commit()
                flash("Signup successful! Please login.")
                return redirect(url_for("login"))
        except Exception as e:
            flash("Signup failed due to server error.")
            print("🚨 Signup Error:", e)
    return render_template("signup.html", datetime=datetime)

from datetime import datetime
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            session["toast"] = f"Welcome back, {username}!"
            return redirect(url_for("predict"))
        else:
            session["toast"] = "Invalid credentials. Try again."
            return redirect(url_for("login"))
    return render_template("login.html", datetime=datetime)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.")
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    history = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.asc()).all()
    detected = sum("Class 1" in h.result for h in history)
    not_detected = sum("Class 2" in h.result for h in history)
    chart_data = [detected, not_detected]

    dates = [h.timestamp.strftime("%Y-%m-%d") for h in history]
    trends = Counter(dates)
    trend_labels = list(trends.keys())
    trend_counts = list(trends.values())

    return render_template(
        "dashboard.html",
        history=history,
        chart_data=chart_data,
        trend_labels=trend_labels,
        trend_counts=trend_counts
    )

@app.route("/admin_dashboard")
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash("Access denied: Admins only.")
        return redirect(url_for("dashboard"))

    users = User.query.all()
    predictions = Prediction.query.order_by(Prediction.timestamp.desc()).all()

    # Build chart data
    class_counts = {"Class 1": 0, "Class 2": 0}
    date_counts = {}

    for p in predictions:
        if "Class 1" in p.result:
            class_counts["Class 1"] += 1
        elif "Class 2" in p.result:
            class_counts["Class 2"] += 1

        day = p.timestamp.strftime("%Y-%m-%d")
        date_counts[day] = date_counts.get(day, 0) + 1

    sorted_dates = sorted(date_counts.items())
    trend_labels = [d[0] for d in sorted_dates]
    trend_counts = [d[1] for d in sorted_dates]

    return render_template(
        "admin_dashboard.html",
        users=users,
        predictions=predictions,
        class_counts=class_counts,
        trend_labels=trend_labels,
        trend_counts=trend_counts
    )

@app.route("/export_csv")
@login_required
def export_csv():
    history = Prediction.query.filter_by(user_id=current_user.id).all()
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(["Timestamp", "Result", "Inputs"])
    for h in history:
        cw.writerow([h.timestamp.strftime("%Y-%m-%d %H:%M"), h.result, h.input_data])
    output = make_response(si.getvalue())
    output.headers['Content-Disposition'] = 'attachment; filename=prediction_history.csv'
    output.headers['Content-type'] = 'text/csv'
    return output

@app.route("/export_all_csv")
@login_required
def export_all_csv():
    if not current_user.is_admin:
        flash("Access denied: Admins only.")
        return redirect(url_for("dashboard"))

    start_str = request.args.get("start")
    end_str = request.args.get("end")

    query = Prediction.query

    if start_str and end_str:
        try:
            start = datetime.strptime(start_str, "%Y-%m-%d")
            end = datetime.strptime(end_str, "%Y-%m-%d")
            end = datetime(end.year, end.month, end.day, 23, 59, 59)  # include full day
            query = query.filter(Prediction.timestamp >= start, Prediction.timestamp <= end)
        except ValueError:
            flash("Invalid date format provided.")
            return redirect(url_for("admin_dashboard"))

    all_predictions = query.order_by(Prediction.timestamp.desc()).all()

    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(["Timestamp", "Username", "Result", "Inputs"])
    for p in all_predictions:
        cw.writerow([p.timestamp.strftime("%Y-%m-%d %H:%M"), p.user.username, p.result, p.input_data])

    output = make_response(si.getvalue())
    output.headers['Content-Disposition'] = 'attachment; filename=filtered_predictions.csv'
    output.headers['Content-type'] = 'text/csv'
    return output

print("🚀 Starting Flask app...")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)