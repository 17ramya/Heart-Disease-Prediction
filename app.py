"""
Heart Disease Prediction - web application
==========================================

A small Flask app that exposes the neural-network models trained in
``train.py`` (loaded from ``model/weights.json`` via the dependency-free
``model.py``) through a browser UI and a JSON API.

Run locally::

    python app.py            # -> http://127.0.0.1:5000

On Vercel the same WSGI ``app`` object is imported by ``api/index.py``.
"""

import os

from flask import Flask, jsonify, render_template, request

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

# --------------------------------------------------------------------------- #
# Feature metadata (drives the HTML form and input validation)
# --------------------------------------------------------------------------- #

FEATURES = [
    {
        "name": "age",
        "label": "Age (years)",
        "type": "number",
        "min": 29,
        "max": 77,
        "step": 1,
        "default": 54,
        "help": "Patient age in years.",
    },
    {
        "name": "sex",
        "label": "Sex",
        "type": "select",
        "default": 1,
        "options": [
            {"value": 0, "label": "Female"},
            {"value": 1, "label": "Male"},
        ],
    },
    {
        "name": "cp",
        "label": "Chest pain type",
        "type": "select",
        "default": 4,
        "options": [
            {"value": 1, "label": "1 - Typical angina"},
            {"value": 2, "label": "2 - Atypical angina"},
            {"value": 3, "label": "3 - Non-anginal pain"},
            {"value": 4, "label": "4 - Asymptomatic"},
        ],
    },
    {
        "name": "trestbps",
        "label": "Resting blood pressure (mm Hg)",
        "type": "number",
        "min": 94,
        "max": 200,
        "step": 1,
        "default": 130,
    },
    {
        "name": "chol",
        "label": "Serum cholesterol (mg/dl)",
        "type": "number",
        "min": 126,
        "max": 564,
        "step": 1,
        "default": 246,
    },
    {
        "name": "fbs",
        "label": "Fasting blood sugar > 120 mg/dl",
        "type": "select",
        "default": 0,
        "options": [
            {"value": 0, "label": "No"},
            {"value": 1, "label": "Yes"},
        ],
    },
    {
        "name": "restecg",
        "label": "Resting electrocardiographic results",
        "type": "select",
        "default": 0,
        "options": [
            {"value": 0, "label": "0 - Normal"},
            {"value": 1, "label": "1 - ST-T wave abnormality"},
            {"value": 2, "label": "2 - Left ventricular hypertrophy"},
        ],
    },
    {
        "name": "thalach",
        "label": "Maximum heart rate achieved",
        "type": "number",
        "min": 71,
        "max": 202,
        "step": 1,
        "default": 150,
    },
    {
        "name": "exang",
        "label": "Exercise induced angina",
        "type": "select",
        "default": 0,
        "options": [
            {"value": 0, "label": "No"},
            {"value": 1, "label": "Yes"},
        ],
    },
    {
        "name": "oldpeak",
        "label": "ST depression induced by exercise (oldpeak)",
        "type": "number",
        "min": 0,
        "max": 6.2,
        "step": 0.1,
        "default": 1.0,
    },
    {
        "name": "slope",
        "label": "Slope of the peak exercise ST segment",
        "type": "select",
        "default": 2,
        "options": [
            {"value": 1, "label": "1 - Upsloping"},
            {"value": 2, "label": "2 - Flat"},
            {"value": 3, "label": "3 - Downsloping"},
        ],
    },
    {
        "name": "ca",
        "label": "Number of major vessels coloured by fluoroscopy",
        "type": "select",
        "default": 0,
        "options": [
            {"value": 0, "label": "0"},
            {"value": 1, "label": "1"},
            {"value": 2, "label": "2"},
            {"value": 3, "label": "3"},
        ],
    },
    {
        "name": "thal",
        "label": "Thalassemia (thal)",
        "type": "select",
        "default": 3,
        "options": [
            {"value": 3, "label": "3 - Normal"},
            {"value": 6, "label": "6 - Fixed defect"},
            {"value": 7, "label": "7 - Reversible defect"},
        ],
    },
]

WEIGHTS_PATH = os.path.join(BASE_DIR, "model", "weights.json")

_model = None


def get_model():
    """Lazily load (and cache) the exported model."""
    global _model
    if _model is None:
        _model = HeartDiseaseModel(WEIGHTS_PATH)
    return _model


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.route("/")
def index():
    return render_template("index.html", features=FEATURES)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/metadata")
def metadata():
    model = get_model()
    return jsonify({"features": FEATURES, "model": model.meta})


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        payload = request.form.to_dict()

    values = []
    for feature in FEATURES:
        raw = payload.get(feature["name"])
        if raw is None or raw == "":
            return jsonify({"error": f"Missing value for '{feature['name']}'."}), 400
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            return (
                jsonify(
                    {
                        "error": (
                            f"Value for '{feature['name']}' must be a number, "
                            f"got {raw!r}."
                        )
                    }
                ),
                400,
            )

    result = get_model().predict(values)
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
