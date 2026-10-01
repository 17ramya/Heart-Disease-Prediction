"""
Heart Disease Prediction - web application
==========================================

A Flask app that serves a rich web UI and a JSON API on top of the models
trained in ``train.py`` (loaded from ``model/weights.json`` via the
dependency-free ``model.py``).

Run locally::

    python app.py            # -> http://127.0.0.1:5000

On Vercel the same WSGI ``app`` object in this module is auto-detected.
"""

import functools
import os
from datetime import timedelta

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

from model import HeartDiseaseModel

# --------------------------------------------------------------------------- #
# App setup
# --------------------------------------------------------------------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.config["JSON_SORT_KEYS"] = False

# --- session / authentication -------------------------------------------------
# On Vercel set SECRET_KEY to a long random string in the project's Environment
# Variables. The fallback keeps local development working out of the box.
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-me")
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Default demo credentials (override with environment variables).
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "heart123"

# --------------------------------------------------------------------------- #
# Feature metadata (drives the form, validation and the reference table)
# --------------------------------------------------------------------------- #

FEATURES = [
    {
        "name": "age",
        "label": "Age",
        "unit": "years",
        "group": "Demographics",
        "type": "number",
        "min": 29, "max": 77, "step": 1, "default": 54,
        "help": "Patient age in years.",
    },
    {
        "name": "sex",
        "label": "Sex",
        "group": "Demographics",
        "type": "select",
        "default": 1,
        "options": [
            {"value": 0, "label": "Female"},
            {"value": 1, "label": "Male"},
        ],
        "help": "Biological sex of the patient.",
    },
    {
        "name": "cp",
        "label": "Chest pain type",
        "group": "Symptoms",
        "type": "select",
        "default": 4,
        "options": [
            {"value": 1, "label": "1 - Typical angina"},
            {"value": 2, "label": "2 - Atypical angina"},
            {"value": 3, "label": "3 - Non-anginal pain"},
            {"value": 4, "label": "4 - Asymptomatic"},
        ],
        "help": "Type of chest pain reported.",
    },
    {
        "name": "exang",
        "label": "Exercise induced angina",
        "group": "Symptoms",
        "type": "select",
        "default": 0,
        "options": [
            {"value": 0, "label": "No"},
            {"value": 1, "label": "Yes"},
        ],
        "help": "Angina brought on by exercise.",
    },
    {
        "name": "trestbps",
        "label": "Resting blood pressure",
        "unit": "mm Hg",
        "group": "Vitals & labs",
        "type": "number",
        "min": 94, "max": 200, "step": 1, "default": 130,
        "help": "Resting blood pressure on admission.",
    },
    {
        "name": "chol",
        "label": "Serum cholesterol",
        "unit": "mg/dl",
        "group": "Vitals & labs",
        "type": "number",
        "min": 126, "max": 564, "step": 1, "default": 246,
        "help": "Serum cholesterol level.",
    },
    {
        "name": "fbs",
        "label": "Fasting blood sugar > 120 mg/dl",
        "group": "Vitals & labs",
        "type": "select",
        "default": 0,
        "options": [
            {"value": 0, "label": "No"},
            {"value": 1, "label": "Yes"},
        ],
        "help": "Fasting blood sugar above 120 mg/dl.",
    },
    {
        "name": "thalach",
        "label": "Maximum heart rate achieved",
        "unit": "bpm",
        "group": "Vitals & labs",
        "type": "number",
        "min": 71, "max": 202, "step": 1, "default": 150,
        "help": "Maximum heart rate reached during a stress test.",
    },
    {
        "name": "restecg",
        "label": "Resting ECG result",
        "group": "ECG & tests",
        "type": "select",
        "default": 0,
        "options": [
            {"value": 0, "label": "0 - Normal"},
            {"value": 1, "label": "1 - ST-T wave abnormality"},
            {"value": 2, "label": "2 - Left ventricular hypertrophy"},
        ],
        "help": "Resting electrocardiographic result.",
    },
    {
        "name": "oldpeak",
        "label": "ST depression (oldpeak)",
        "group": "ECG & tests",
        "type": "number",
        "min": 0, "max": 6.2, "step": 0.1, "default": 1.0,
        "help": "ST depression induced by exercise relative to rest.",
    },
    {
        "name": "slope",
        "label": "Slope of peak exercise ST segment",
        "group": "ECG & tests",
        "type": "select",
        "default": 2,
        "options": [
            {"value": 1, "label": "1 - Upsloping"},
            {"value": 2, "label": "2 - Flat"},
            {"value": 3, "label": "3 - Downsloping"},
        ],
        "help": "Shape of the ST segment at peak exercise.",
    },
    {
        "name": "ca",
        "label": "Major vessels coloured by fluoroscopy",
        "group": "ECG & tests",
        "type": "select",
        "default": 0,
        "options": [
            {"value": 0, "label": "0"},
            {"value": 1, "label": "1"},
            {"value": 2, "label": "2"},
            {"value": 3, "label": "3"},
        ],
        "help": "Number of major vessels (0-3) seen via fluoroscopy.",
    },
    {
        "name": "thal",
        "label": "Thalassemia (thal)",
        "group": "ECG & tests",
        "type": "select",
        "default": 3,
        "options": [
            {"value": 3, "label": "3 - Normal"},
            {"value": 6, "label": "6 - Fixed defect"},
            {"value": 7, "label": "7 - Reversible defect"},
        ],
        "help": "Thalassemia / thallium stress test result.",
    },
]

#: Groups used to lay the form out into collapsible sections.
FEATURE_GROUPS = ["Demographics", "Symptoms", "Vitals & labs", "ECG & tests"]

WEIGHTS_PATH = os.path.join(BASE_DIR, "model", "weights.json")

_model = None


def get_model():
    """Lazily load (and cache) the exported models."""
    global _model
    if _model is None:
        _model = HeartDiseaseModel(WEIGHTS_PATH)
    return _model


def merged_features():
    """Feature definitions enriched with dataset statistics for the UI."""
    stats = get_model().feature_stats
    enriched = []
    for feature in FEATURES:
        item = dict(feature)
        item["stats"] = stats.get(feature["name"], {})
        # Hint shown in the (empty) number inputs - the mean of the dataset.
        mean = item["stats"].get("mean")
        item["placeholder"] = f"e.g. {mean}" if mean is not None else "Enter value"
        enriched.append(item)
    return enriched


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #


def _load_users():
    """Load users from the environment.

    * ``APP_USERS``     - comma separated ``user:password`` pairs, e.g.
      ``"alice:secret,bob:hunter2"``.
    * ``APP_USERNAME`` / ``APP_PASSWORD`` - a single account.

    Falls back to the built-in demo account when nothing is configured.
    Passwords are stored as salted hashes.
    """
    users = {}
    raw = os.environ.get("APP_USERS", "").strip()
    if raw:
        for pair in raw.split(","):
            pair = pair.strip()
            if ":" in pair:
                username, password = pair.split(":", 1)
                username = username.strip()
                if username:
                    users[username] = generate_password_hash(password)

    if not users:
        username = os.environ.get("APP_USERNAME", DEFAULT_USERNAME).strip()
        password = os.environ.get("APP_PASSWORD", DEFAULT_PASSWORD)
        users[username or DEFAULT_USERNAME] = generate_password_hash(password)
    return users


USERS = _load_users()


def authenticate(username, password):
    """Return the username when the credentials are valid, else ``None``."""
    stored = USERS.get(username)
    if stored and check_password_hash(stored, password):
        return username
    return None


def login_required(view):
    """Protect a view: API routes answer 401, pages redirect to /login."""

    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "Authentication required."}), 401
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def _safe_next(target):
    """Only allow same-site relative redirect targets."""
    if target and target.startswith("/") and not target.startswith("//"):
        return target
    return None


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user"):
        return redirect(_safe_next(request.args.get("next")) or url_for("index"))

    error = None
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        if authenticate(username, password):
            session.clear()
            session.permanent = True
            session["user"] = username
            target = _safe_next(request.args.get("next")) or _safe_next(
                request.form.get("next")
            )
            return redirect(target or url_for("index"))
        error = "Incorrect username or password."

    return render_template(
        "login.html", error=error, next=request.args.get("next", "")
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/api/session")
def session_info():
    user = session.get("user")
    return jsonify({"authenticated": bool(user), "user": user})


@app.route("/")
@login_required
def index():
    model = get_model()
    return render_template(
        "index.html",
        features=merged_features(),
        feature_groups=FEATURE_GROUPS,
        meta=model.meta,
        current_user=session.get("user"),
    )


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/metadata")
@login_required
def metadata():
    model = get_model()
    return jsonify(
        {
            "features": merged_features(),
            "groups": FEATURE_GROUPS,
            "model": model.meta,
        }
    )


@app.route("/api/predict", methods=["POST"])
@login_required
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        payload = request.form.to_dict()

    model = get_model()

    # Build the feature vector in the *model's* column order (the dataset order),
    # which may differ from the display order used by the form.
    values = []
    for name in model.features:
        raw = payload.get(name)
        if raw is None or raw == "":
            return jsonify({"error": f"Missing value for '{name}'."}), 400
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            return (
                jsonify({"error": f"Value for '{name}' must be a number, got {raw!r}."}),
                400,
            )

    try:
        result = model.predict(values)
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": f"Prediction failed: {exc}"}), 500
    return jsonify(result)


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(_error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

