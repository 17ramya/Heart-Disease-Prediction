"""
Light-weight, dependency-free inference for the heart-disease models.

Every model trained by ``train.py`` is exported to ``model/weights.json`` and
re-implemented here in **pure Python** (standard library only). The web app
therefore needs neither TensorFlow nor scikit-learn at runtime, which keeps it
small enough to deploy inside a Vercel serverless function.

Models
------
* Neural network (binary)      - relu -> relu -> sigmoid
* Neural network (categorical) - relu -> relu -> softmax (5 classes)
* Logistic Regression          - standardised linear model
* Gaussian Naive Bayes
* K-Nearest Neighbours
* Random Forest                - tree traversal + averaging
* Ensemble                     - mean of the binary models' probabilities
"""

import json
import math
import os

# --------------------------------------------------------------------------- #
# Small maths helpers
# --------------------------------------------------------------------------- #


def _relu(x):
    return x if x > 0.0 else 0.0


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _softmax(values):
    largest = max(values)
    exps = [math.exp(v - largest) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


_ACTIVATIONS = {"relu": _relu, "sigmoid": _sigmoid, "softmax": _softmax}

#: Order in which the binary models are shown in the UI.
MODEL_ORDER = ["nn_binary", "logistic", "random_forest", "gaussian_nb", "knn"]


class HeartDiseaseModel:
    """Loads ``weights.json`` and runs predictions with the standard library."""

    def __init__(self, weights_path=None):
        if weights_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            weights_path = os.path.join(base, "model", "weights.json")

        self.weights_path = weights_path
        with open(weights_path, "r", encoding="utf-8") as fh:
            self.data = json.load(fh)

        self.meta = self.data.get("meta", {})
        self.features = self.meta.get("features", [])
        self.class_labels = self.meta.get(
            "class_labels", [str(i) for i in self.meta.get("classes", [])]
        )
        self.feature_stats = self.meta.get("feature_stats", {})
        self.model_info = self.meta.get("models", {})

    # ------------------------------------------------------------------ #
    # Individual models
    # ------------------------------------------------------------------ #

    def _nn_forward(self, layers, inputs):
        activations = list(inputs)
        for layer in layers:
            kernel, bias = layer["kernel"], layer["bias"]
            activation = _ACTIVATIONS[layer["activation"]]
            outputs = []
            for j, b in enumerate(bias):
                total = b
                for i, value in enumerate(activations):
                    total += value * kernel[i][j]
                outputs.append(total)
            if activation is _softmax:
                activations = activation(outputs)
            else:
                activations = [activation(o) for o in outputs]
        return activations

    def _nn_binary_proba(self, x):
        return float(self._nn_forward(self.data["nn_binary"]["layers"], x)[0])

    def _logistic_proba(self, x):
        d = self.data["logistic"]
        z = d["intercept"]
        for i, value in enumerate(x):
            z += d["coef"][i] * (value - d["mean"][i]) / d["scale"][i]
        return _sigmoid(z)

    def _gaussian_nb_proba(self, x):
        d = self.data["gaussian_nb"]
        theta, var, prior = d["theta"], d["var"], d["prior"]
        log_posteriors = []
        for c in range(len(theta)):
            ll = math.log(prior[c])
            for i, value in enumerate(x):
                ll += -0.5 * math.log(2 * math.pi * var[c][i])
                ll += -((value - theta[c][i]) ** 2) / (2 * var[c][i])
            log_posteriors.append(ll)
        probs = _softmax(log_posteriors)
        classes = d["classes"]
        idx = classes.index(1) if 1 in classes else len(classes) - 1
        return probs[idx]

    def _knn_proba(self, x):
        d = self.data["knn"]
        train_x, train_y, k = d["X"], d["y"], d["k"]
        neighbours = []
        for i, row in enumerate(train_x):
            dist = 0.0
            for j, value in enumerate(x):
                diff = row[j] - value
                dist += diff * diff
            neighbours.append((dist, train_y[i]))
        neighbours.sort(key=lambda pair: pair[0])
        nearest = neighbours[:k]
        return sum(label for _, label in nearest) / float(len(nearest))

    def _forest_proba(self, x):
        trees = self.data["random_forest"]["trees"]
        total = 0.0
        for tree in trees:
            node = 0
            while tree["children_left"][node] != -1:
                feature = tree["feature"][node]
                if x[feature] <= tree["threshold"][node]:
                    node = tree["children_left"][node]
                else:
                    node = tree["children_right"][node]
            total += tree["prob"][node]
        return total / float(len(trees))

    def _categorical_proba(self, x):
        layers = self.data["nn_categorical"]["layers"]
        return [float(p) for p in self._nn_forward(layers, x)]

    # ------------------------------------------------------------------ #
    # Aggregation helpers
    # ------------------------------------------------------------------ #

    def _contributions(self, x, top=6):
        """Logistic-regression risk contributions (standardised value * coef)."""
        d = self.data["logistic"]
        contributions = []
        for i, name in enumerate(self.features):
            value = d["coef"][i] * (x[i] - d["mean"][i]) / d["scale"][i]
            contributions.append(
                {
                    "feature": name,
                    "contribution": round(value, 4),
                    "increases_risk": value > 0,
                    "raw_value": x[i],
                }
            )
        contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)
        return contributions[:top]

    def _model_label(self, key):
        return self.model_info.get(key, {}).get("label", key)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def predict(self, features):
        """Run every model and return a rich, JSON-serialisable result."""
        x = [float(v) for v in features]

        probabilities = {
            "nn_binary": self._nn_binary_proba(x),
            "logistic": self._logistic_proba(x),
            "gaussian_nb": self._gaussian_nb_proba(x),
            "knn": self._knn_proba(x),
            "random_forest": self._forest_proba(x),
        }
        ensemble_probability = sum(probabilities.values()) / len(probabilities)

        models = []
        for key in MODEL_ORDER:
            probability = probabilities[key]
            info = self.model_info.get(key, {})
            models.append(
                {
                    "key": key,
                    "label": self._model_label(key),
                    "probability": round(probability, 4),
                    "prediction": (
                        "Heart Disease" if probability >= 0.5 else "No Heart Disease"
                    ),
                    "accuracy": info.get("accuracy"),
                    "roc_auc": info.get("roc_auc"),
                }
            )

        categorical = self._categorical_proba(x)
        class_index = max(range(len(categorical)), key=categorical.__getitem__)

        def _label_for(i):
            return self.class_labels[i] if i < len(self.class_labels) else str(i)

        binary_label = 1 if ensemble_probability >= 0.5 else 0

        return {
            "input": dict(zip(self.features, x)),
            "binary": {
                "prediction": "Heart Disease" if binary_label else "No Heart Disease",
                "label": binary_label,
                "probability": round(ensemble_probability, 4),
                "confidence": round(
                    ensemble_probability if binary_label else 1 - ensemble_probability,
                    4,
                ),
            },
            "models": models,
            "categorical": {
                "class": class_index,
                "class_label": _label_for(class_index),
                "probabilities": {
                    _label_for(i): round(p, 4) for i, p in enumerate(categorical)
                },
            },
            "risk_factors": self._contributions(x),
        }

