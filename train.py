"""
Advancing Heart Disease Prediction: Neural Network Models for Accurate Diagnosis
================================================================================

This file is the plain-Python conversion of the original Jupyter notebook
``Heart disease prediction/Heart Disease Prediction.ipynb``.

It performs the exact same steps as the notebook:

    1. Download and load the UCI Cleveland heart disease dataset.
    2. Clean the missing values (marked with "?").
    3. Split the data into training / testing sets.
    4. Train a *categorical* (5-class) neural network.
    5. Train a *binary*    (disease / no-disease) neural network.
    6. Evaluate both models.
    7. *** NEW *** Export the trained weights to ``model/weights.json`` so that
       the light-weight web application (``app.py``) can serve predictions
       **without** needing TensorFlow at runtime. This is what makes the app
       deployable on Vercel, where a full TensorFlow install is too large.

Run it with::

    python train.py

Requirements are listed in ``requirements-train.txt``.
"""

import json
import os
import ssl
import urllib.request
from datetime import datetime, timezone
from io import StringIO

import numpy as np

# Matplotlib is only used by the notebook to draw the histograms. We keep the
# same behaviour but make sure it never blocks on a GUI when run from a script.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pandas.plotting import scatter_matrix  # noqa: E402  (kept for parity)

import pandas as pd  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: UCI Cleveland dataset (same URL the notebook uses).
DATA_URL = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

#: Optional local copy of the dataset. If this file exists it is used instead
#: of downloading, which is handy when offline.
LOCAL_DATA_PATH = os.path.join("data", "processed.cleveland.data")

#: Column names for the dataset.
NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "class",
]

#: Where the exported weights are written (consumed by ``app.py``).
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "weights.json")

#: Reproducibility. The notebook used ``train_test_split`` without a seed, so
#: every run produced a different split. We fix the seed here so the exported
#: model is deterministic (a small, deliberate production improvement).
RANDOM_STATE = 42

#: Human friendly names for the 5 severity levels of the original dataset.
CLASS_LABELS = [
    "No disease (0)",
    "Mild (1)",
    "Moderate (2)",
    "Severe (3)",
    "Very severe (4)",
]


# --------------------------------------------------------------------------- #
# 1. Importing the dataset
# --------------------------------------------------------------------------- #

def load_data():
    """Download (or load locally) the Cleveland heart disease dataset."""
    if os.path.exists(LOCAL_DATA_PATH):
        print(f"Loading local dataset from {LOCAL_DATA_PATH!r} ...")
        return pd.read_csv(LOCAL_DATA_PATH, names=NAMES)

    print(f"Downloading dataset from {DATA_URL} ...")
    try:
        return pd.read_csv(DATA_URL, names=NAMES)
    except Exception:  # pragma: no cover - network/SSL fallback
        # Some environments have an out-of-date certificate store; retry with a
        # relaxed context before giving up.
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(DATA_URL, context=context, timeout=60) as resp:
            payload = resp.read().decode("utf-8")
        return pd.read_csv(StringIO(payload), names=NAMES)


def clean_data(cleveland):
    """Replicate the notebook's cleaning steps.

    The notebook first turns every ``"?"`` into ``NaN`` (via the element-wise
    ``df[mask]`` selection) and then drops any row that contains a ``NaN``.
    """
    data = cleveland[~cleveland.isin(["?"])]
    data = data.dropna(axis=0)
    data = data.apply(pd.to_numeric)
    return data


# --------------------------------------------------------------------------- #
# 2. Training / testing datasets
# --------------------------------------------------------------------------- #

def split_data(data):
    from sklearn import model_selection

    X = np.array(data.drop(columns=["class"]))
    y = np.array(data["class"])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# 3. Categorical (5-class) model
# --------------------------------------------------------------------------- #

def create_model():
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.optimizers import Adam

    model = Sequential()
    model.add(Dense(8, input_dim=13, kernel_initializer="normal", activation="relu"))
    model.add(Dense(4, kernel_initializer="normal", activation="relu"))
    model.add(Dense(5, activation="softmax"))

    adam = Adam(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
    return model


# --------------------------------------------------------------------------- #
# 4. Binary model
# --------------------------------------------------------------------------- #

def create_binary_model():
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.optimizers import Adam

    model = Sequential()
    model.add(Dense(8, input_dim=13, kernel_initializer="normal", activation="relu"))
    model.add(Dense(4, kernel_initializer="normal", activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    adam = Adam(learning_rate=0.0001)
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])
    return model


# --------------------------------------------------------------------------- #
# 5. Export
# --------------------------------------------------------------------------- #

def layer_to_dict(weights, activation):
    """Convert a Keras dense layer ``[kernel, bias]`` into plain Python lists."""
    kernel, bias = weights
    return {
        "activation": activation,
        "kernel": np.asarray(kernel).tolist(),
        "bias": np.asarray(bias).tolist(),
    }


def export_weights(model, binary_model, categorical_acc, binary_acc, n_train):
    """Write the trained weights + metadata to ``model/weights.json``."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    payload = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "features": NAMES[:-1],
            "classes": [0, 1, 2, 3, 4],
            "class_labels": CLASS_LABELS,
            "n_training_samples": int(n_train),
            "categorical_accuracy": round(float(categorical_acc), 4),
            "binary_accuracy": round(float(binary_acc), 4),
        },
        "categorical": {
            "layers": [
                layer_to_dict(model.get_weights()[0:2], "relu"),
                layer_to_dict(model.get_weights()[2:4], "relu"),
                layer_to_dict(model.get_weights()[4:6], "softmax"),
            ]
        },
        "binary": {
            "layers": [
                layer_to_dict(binary_model.get_weights()[0:2], "relu"),
                layer_to_dict(binary_model.get_weights()[2:4], "relu"),
                layer_to_dict(binary_model.get_weights()[4:6], "sigmoid"),
            ]
        },
    }

    with open(WEIGHTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nExported trained weights -> {WEIGHTS_PATH}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    print("=" * 70)
    print("Advancing Heart Disease Prediction - Neural Network Models")
    print("=" * 70)

    # ---- 1. Importing dataset -------------------------------------------------
    cleveland = load_data()
    print(f"\nRaw dataset shape: {cleveland.shape}")
    print(cleveland.loc[1])

    data = clean_data(cleveland)
    print(f"\nAfter cleaning (missing values removed): {data.shape}")
    print(data.dtypes)

    # Exploratory plot (saved to disk, matching the notebook's histograms).
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        data.hist(figsize=(12, 12))
        plot_path = os.path.join(MODEL_DIR, "feature_histograms.png")
        plt.savefig(plot_path)
        plt.close("all")
        print(f"Saved feature histograms -> {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"(Skipping histogram plot: {exc})")

    # ---- 2. Training / testing datasets --------------------------------------
    from keras.utils import to_categorical

    X_train, X_test, y_train, y_test = split_data(data)

    Y_train = to_categorical(y_train, num_classes=None)
    print(f"\nY_train shape: {Y_train.shape}")

    # ---- 3. Categorical model ------------------------------------------------
    print("\n----- TRAINING: CATEGORICAL MODEL -----")
    model = create_model()
    model.summary()
    model.fit(X_train, Y_train, epochs=100, batch_size=10, verbose=1)

    # ---- 4. Binary model -----------------------------------------------------
    print("\n----- TRAINING: BINARY MODEL -----")
    Y_train_binary = y_train.copy()
    Y_test_binary = y_test.copy()
    Y_train_binary[Y_train_binary > 0] = 1
    Y_test_binary[Y_test_binary > 0] = 1

    binary_model = create_binary_model()
    binary_model.summary()
    binary_model.fit(X_train, Y_train_binary, epochs=100, batch_size=10, verbose=1)

    # ---- 5. Evaluation -------------------------------------------------------
    from sklearn.metrics import accuracy_score, classification_report

    categorical_pred = np.argmax(model.predict(X_test), axis=1)
    categorical_acc = accuracy_score(y_test, categorical_pred)
    print("\nResults for Categorical Model")
    print(categorical_acc)
    print(classification_report(y_test, categorical_pred))

    binary_pred = np.round(binary_model.predict(X_test)).astype(int)
    binary_acc = accuracy_score(Y_test_binary, binary_pred)
    print("Results for Binary Model")
    print(binary_acc)
    print(classification_report(Y_test_binary, binary_pred))

    # ---- 6. Export -----------------------------------------------------------
    export_weights(model, binary_model, categorical_acc, binary_acc, len(X_train))
    print("\nDone. You can now run the web app with:  python app.py")


if __name__ == "__main__":
    main()

