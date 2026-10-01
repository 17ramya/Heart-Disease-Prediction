"""
Advancing Heart Disease Prediction: Neural Network Models for Accurate Diagnosis
================================================================================

Plain-Python conversion of the original Jupyter notebook
(``Heart disease prediction/Heart Disease Prediction.ipynb``), extended so the
web application can offer **multiple prediction models**.

Steps
-----
1. Download / load the UCI Cleveland heart disease dataset.
2. Clean missing values (marked with "?").
3. Split into train / test sets.
4. Train the two neural networks from the notebook:
       * categorical : 13 -> 8 -> 4 -> 5 (softmax)
       * binary      : 13 -> 8 -> 4 -> 1 (sigmoid)
5. Train additional classic models (binary: disease / no-disease):
       * Logistic Regression
       * Gaussian Naive Bayes
       * K-Nearest Neighbours
       * Random Forest
6. Evaluate every model on the hold-out test set.
7. Export **everything** to ``model/weights.json`` in a format that the
   dependency-free ``model.py`` can evaluate in pure Python.  This is what lets
   the web app run on Vercel without TensorFlow or scikit-learn installed.

Run with::

    python train.py

Requirements: ``requirements-train.txt``.
"""

import json
import os
import random
import ssl
import urllib.request
from datetime import datetime, timezone
from io import StringIO

import numpy as np

# Matplotlib is only used for the exploratory histograms.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pandas.plotting import scatter_matrix  # noqa: E402  (kept for parity)

import pandas as pd  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DATA_URL = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
LOCAL_DATA_PATH = os.path.join("data", "processed.cleveland.data")

NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "class",
]

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "weights.json")

RANDOM_STATE = 42

CLASS_LABELS = [
    "No disease (0)",
    "Mild (1)",
    "Moderate (2)",
    "Severe (3)",
    "Very severe (4)",
]


# --------------------------------------------------------------------------- #
# 1-2. Data loading & cleaning
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
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(DATA_URL, context=context, timeout=60) as resp:
            payload = resp.read().decode("utf-8")
        return pd.read_csv(StringIO(payload), names=NAMES)


def clean_data(cleveland):
    """Replicate the notebook's cleaning steps (? -> NaN, then drop rows)."""
    data = cleveland[~cleveland.isin(["?"])]
    data = data.dropna(axis=0)
    data = data.apply(pd.to_numeric)
    return data


def split_data(data):
    from sklearn import model_selection

    X = np.array(data.drop(columns=["class"]))
    y = np.array(data["class"])
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def feature_stats(data):
    """Per-feature descriptive statistics (min / max / mean / std)."""
    stats = {}
    for name in NAMES[:-1]:
        column = data[name]
        stats[name] = {
            "min": round(float(column.min()), 2),
            "max": round(float(column.max()), 2),
            "mean": round(float(column.mean()), 2),
            "std": round(float(column.std()), 2),
        }
    return stats


# --------------------------------------------------------------------------- #
# 3. Neural networks (from the notebook)
# --------------------------------------------------------------------------- #

def build_categorical_nn():
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.optimizers import Adam

    model = Sequential()
    model.add(Dense(8, input_dim=13, kernel_initializer="normal", activation="relu"))
    model.add(Dense(4, kernel_initializer="normal", activation="relu"))
    model.add(Dense(5, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return model


def build_binary_nn():
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.optimizers import Adam

    model = Sequential()
    model.add(Dense(8, input_dim=13, kernel_initializer="normal", activation="relu"))
    model.add(Dense(4, kernel_initializer="normal", activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.0001),
        metrics=["accuracy"],
    )
    return model



# --------------------------------------------------------------------------- #
# 4. Classic models (scikit-learn)
# --------------------------------------------------------------------------- #

def train_classic_models(X_train, y_train_binary):
    """Fit Logistic Regression, Gaussian NB, KNN and a Random Forest."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train)
    scaled = scaler.transform(X_train)

    logistic = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    logistic.fit(scaled, y_train_binary)

    nb = GaussianNB().fit(X_train, y_train_binary)

    knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train_binary)

    forest = RandomForestClassifier(
        n_estimators=60, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1
    ).fit(X_train, y_train_binary)

    return {"scaler": scaler, "logistic": logistic, "nb": nb,
            "knn": knn, "forest": forest}


def _tree_to_dict(estimator):
    """Serialise a fitted DecisionTreeClassifier to plain lists."""
    tree = estimator.tree_
    totals = tree.value[:, 0, :].sum(axis=1)
    prob = [
        float(tree.value[i, 0, 1] / totals[i]) if totals[i] > 0 else 0.0
        for i in range(len(totals))
    ]
    return {
        "children_left": tree.children_left.tolist(),
        "children_right": tree.children_right.tolist(),
        "feature": tree.feature.tolist(),
        "threshold": [float(t) for t in tree.threshold],
        "prob": prob,
    }


def export_classic(models):
    """Convert the fitted classic models into JSON-serialisable dicts."""
    scaler, logistic = models["scaler"], models["logistic"]
    nb, knn, forest = models["nb"], models["knn"], models["forest"]

    return {
        "logistic": {
            "type": "logistic",
            "mean": [float(v) for v in scaler.mean_],
            "scale": [float(v) for v in scaler.scale_],
            "coef": [float(v) for v in logistic.coef_[0]],
            "intercept": float(logistic.intercept_[0]),
        },
        "gaussian_nb": {
            "type": "gaussian_nb",
            "classes": [int(c) for c in nb.classes_],
            "theta": [[float(v) for v in row] for row in nb.theta_],
            "var": [[float(v) for v in row] for row in nb.var_],
            "prior": [float(v) for v in nb.class_prior_],
        },
        "knn": {
            "type": "knn",
            "k": int(knn.n_neighbors),
            "X": [[float(v) for v in row] for row in knn._fit_X],
            "y": [int(knn.classes_[i]) for i in knn._y],
        },
        "random_forest": {
            "type": "random_forest",
            "trees": [_tree_to_dict(est) for est in forest.estimators_],
        },
    }


def evaluate_binary(probabilities, y_true):
    """Accuracy + ROC-AUC for a 0/1 classifier."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    preds = [1 if p >= 0.5 else 0 for p in probabilities]
    accuracy = accuracy_score(y_true, preds)
    try:
        auc = roc_auc_score(y_true, probabilities)
    except ValueError:  # only one class present in y_true
        auc = float("nan")
    return round(float(accuracy), 4), round(float(auc), 4)



# --------------------------------------------------------------------------- #
# 5. Weight export
# --------------------------------------------------------------------------- #

def _nn_layers(model):
    """Serialise the 3 dense layers of a Keras model to plain lists."""
    weights = model.get_weights()
    activations = ["relu", "relu", "softmax" if weights[4].shape[-1] > 1 else "sigmoid"]
    layers = []
    for i, activation in enumerate(activations):
        kernel, bias = weights[2 * i], weights[2 * i + 1]
        layers.append(
            {
                "activation": activation,
                "kernel": np.asarray(kernel).tolist(),
                "bias": np.asarray(bias).tolist(),
            }
        )
    return layers


def _predict_proba_classic(exported, X):
    """Get class-1 probabilities from the exported classic models."""
    import numpy as _np

    scaler_mean = _np.array(exported["logistic"]["mean"])
    scaler_scale = _np.array(exported["logistic"]["scale"])
    coef = _np.array(exported["logistic"]["coef"])
    intercept = exported["logistic"]["intercept"]
    scaled = (X - scaler_mean) / scaler_scale
    logistic_p = 1.0 / (1.0 + _np.exp(-(scaled @ coef + intercept)))

    # Gaussian NB
    theta = _np.array(exported["gaussian_nb"]["theta"])
    var = _np.array(exported["gaussian_nb"]["var"])
    prior = _np.array(exported["gaussian_nb"]["prior"])
    log_likelihood = []
    for c in range(theta.shape[0]):
        ll = -0.5 * _np.sum(
            _np.log(2 * _np.pi * var[c]) + ((X - theta[c]) ** 2) / var[c], axis=1
        )
        log_likelihood.append(_np.log(prior[c]) + ll)
    log_likelihood = _np.vstack(log_likelihood).T
    nb_p = _np.exp(log_likelihood - log_likelihood.max(axis=1, keepdims=True))
    nb_p = nb_p / nb_p.sum(axis=1, keepdims=True)
    nb_p = nb_p[:, 1]

    # KNN
    train_X = _np.array(exported["knn"]["X"])
    train_y = _np.array(exported["knn"]["y"])
    k = exported["knn"]["k"]
    knn_p = []
    for row in X:
        distances = _np.sqrt(((train_X - row) ** 2).sum(axis=1))
        nearest = train_y[_np.argsort(distances)[:k]]
        knn_p.append(float(nearest.mean()))

    # Random forest
    forest_p = []
    for row in X:
        probs = []
        for tree in exported["random_forest"]["trees"]:
            node = 0
            while tree["children_left"][node] != -1:
                if row[tree["feature"][node]] <= tree["threshold"][node]:
                    node = tree["children_left"][node]
                else:
                    node = tree["children_right"][node]
            probs.append(tree["prob"][node])
        forest_p.append(float(sum(probs) / len(probs)))

    return {
        "logistic": [float(p) for p in logistic_p],
        "gaussian_nb": [float(p) for p in nb_p],
        "knn": knn_p,
        "random_forest": forest_p,
    }


def build_payload(nn_cat, nn_bin, exported, X_test, y_test_binary,
                  categorical_acc, data):
    """Assemble the complete ``weights.json`` payload."""
    classic_probs = _predict_proba_classic(exported, X_test)

    nn_bin_probs = nn_bin.predict(X_test, verbose=0).ravel().tolist()
    logistic_acc, logistic_auc = evaluate_binary(classic_probs["logistic"], y_test_binary)
    nb_acc, nb_auc = evaluate_binary(classic_probs["gaussian_nb"], y_test_binary)
    knn_acc, knn_auc = evaluate_binary(classic_probs["knn"], y_test_binary)
    forest_acc, forest_auc = evaluate_binary(classic_probs["random_forest"], y_test_binary)
    nn_acc, nn_auc = evaluate_binary(nn_bin_probs, y_test_binary)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "features": NAMES[:-1],
        "classes": [0, 1, 2, 3, 4],
        "class_labels": CLASS_LABELS,
        "n_training_samples": int(len(data.index) - len(X_test)),
        "n_test_samples": int(len(X_test)),
        "n_total_samples": int(len(data.index)),
        "categorical_accuracy": round(float(categorical_acc), 4),
        "feature_stats": feature_stats(data),
        "models": {
            "nn_binary": {"label": "Neural Network", "accuracy": nn_acc, "roc_auc": nn_auc},
            "logistic": {"label": "Logistic Regression", "accuracy": logistic_acc, "roc_auc": logistic_auc},
            "gaussian_nb": {"label": "Gaussian Naive Bayes", "accuracy": nb_acc, "roc_auc": nb_auc},
            "knn": {"label": "K-Nearest Neighbours", "accuracy": knn_acc, "roc_auc": knn_auc},
            "random_forest": {"label": "Random Forest", "accuracy": forest_acc, "roc_auc": forest_auc},
        },
    }

    payload = {
        "meta": meta,
        "nn_binary": {"type": "neural_network", "layers": _nn_layers(nn_bin)},
        "nn_categorical": {"type": "neural_network", "layers": _nn_layers(nn_cat)},
    }
    payload.update(exported)
    return payload




# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    print("=" * 70)
    print("Advancing Heart Disease Prediction - Neural Network Models")
    print("=" * 70)

    # ---- data ----------------------------------------------------------------
    cleveland = load_data()
    print(f"\nRaw dataset shape: {cleveland.shape}")
    print(cleveland.loc[1])

    data = clean_data(cleveland)
    print(f"\nAfter cleaning (missing values removed): {data.shape}")

    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        data.hist(figsize=(12, 12))
        plt.savefig(os.path.join(MODEL_DIR, "feature_histograms.png"))
        plt.close("all")
        print(f"Saved feature histograms -> {os.path.join(MODEL_DIR, 'feature_histograms.png')}")
    except Exception as exc:  # pragma: no cover
        print(f"(Skipping histogram plot: {exc})")

    try:  # reproducible Keras training
        from keras.utils import set_random_seed

        set_random_seed(RANDOM_STATE)
    except Exception:  # pragma: no cover
        pass

    from keras.utils import to_categorical
    from sklearn.metrics import accuracy_score, classification_report

    X_train, X_test, y_train, y_test = split_data(data)
    Y_train = to_categorical(y_train, num_classes=None)

    y_train_binary = y_train.copy()
    y_test_binary = y_test.copy()
    y_train_binary[y_train_binary > 0] = 1
    y_test_binary[y_test_binary > 0] = 1

    # ---- categorical neural network -----------------------------------------
    print("\n----- TRAINING: CATEGORICAL NEURAL NETWORK -----")
    nn_cat = build_categorical_nn()
    nn_cat.summary()
    nn_cat.fit(X_train, Y_train, epochs=100, batch_size=10, verbose=1)
    cat_pred = np.argmax(nn_cat.predict(X_test, verbose=0), axis=1)
    categorical_acc = accuracy_score(y_test, cat_pred)
    print(f"\nCategorical accuracy: {categorical_acc:.4f}")
    print(classification_report(y_test, cat_pred))

    # ---- binary neural network ----------------------------------------------
    print("\n----- TRAINING: BINARY NEURAL NETWORK -----")
    nn_bin = build_binary_nn()
    nn_bin.summary()
    nn_bin.fit(X_train, y_train_binary, epochs=100, batch_size=10, verbose=1)
    bin_pred = np.round(nn_bin.predict(X_test, verbose=0)).astype(int).ravel()
    print(f"\nBinary accuracy: {accuracy_score(y_test_binary, bin_pred):.4f}")
    print(classification_report(y_test_binary, bin_pred))

    # ---- classic models ------------------------------------------------------
    print("\n----- TRAINING: CLASSIC MODELS -----")
    classic = train_classic_models(X_train, y_train_binary)
    exported = export_classic(classic)

    # ---- export --------------------------------------------------------------
    payload = build_payload(
        nn_cat, nn_bin, exported, X_test, y_test_binary, categorical_acc, data
    )
    with open(WEIGHTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("\nModel comparison (hold-out test set):")
    for info in payload["meta"]["models"].values():
        print(
            f"  {info['label']:<26} accuracy={info['accuracy']:.3f} "
            f"roc_auc={info['roc_auc']:.3f}"
        )
    print(f"\nExported trained weights -> {WEIGHTS_PATH}")
    print("Run the web app with:  python app.py")


if __name__ == "__main__":
    main()
