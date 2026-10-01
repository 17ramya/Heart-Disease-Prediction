"""
Light-weight inference for the heart-disease neural networks.

This module performs a *pure-Python* forward pass over the weights exported by
``train.py`` (``model/weights.json``).  Because it needs **no** TensorFlow /
Keras / NumPy at runtime, it is small enough to deploy inside a Vercel
serverless function (TensorFlow alone is bigger than Vercel's 250 MB limit).

The maths is identical to the Keras models:

    categorical : relu -> relu -> softmax   (5 classes)
    binary      : relu -> relu -> sigmoid   (heart disease / no heart disease)
"""

import json
import math
import os


# --------------------------------------------------------------------------- #
# Activation functions
# --------------------------------------------------------------------------- #

def _relu(x):
    return x if x > 0.0 else 0.0


def _sigmoid(x):
    # Numerically stable for very large |x|.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _softmax(values):
    largest = max(values)
    exps = [math.exp(v - largest) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


_ACTIVATIONS = {
    "relu": _relu,
    "sigmoid": _sigmoid,
    "softmax": _softmax,
}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class HeartDiseaseModel:
    """Loads ``weights.json`` and runs predictions in plain Python."""

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
        self._categorical_layers = self.data["categorical"]["layers"]
        self._binary_layers = self.data["binary"]["layers"]

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _forward(layers, inputs):
        activations = list(inputs)
        for layer in layers:
            kernel = layer["kernel"]
            bias = layer["bias"]
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

    # -- public API ------------------------------------------------------------

    def predict_binary(self, features):
        """Return ``(probability_of_disease, predicted_label)``."""
        probability = self._forward(self._binary_layers, features)
        probability = float(probability[0])
        label = 1 if probability >= 0.5 else 0
        return probability, label

    def predict_categorical(self, features):
        """Return the 5-class probability vector and the arg-max class."""
        probabilities = self._forward(self._categorical_layers, features)
        probabilities = [float(p) for p in probabilities]
        index = max(range(len(probabilities)), key=probabilities.__getitem__)
        return probabilities, index

    def predict(self, features):
        """Run both models and return a JSON-serialisable result dict."""
        features = [float(v) for v in features]

        probabilities, index = self.predict_categorical(features)
        binary_probability, binary_label = self.predict_binary(features)

        return {
            "input": dict(zip(self.features, features)) if self.features else features,
            "binary": {
                "prediction": "Heart Disease" if binary_label == 1 else "No Heart Disease",
                "label": binary_label,
                "probability": round(binary_probability, 4),
                "confidence": round(
                    binary_probability if binary_label == 1 else 1 - binary_probability,
                    4,
                ),
            },
            "categorical": {
                "class": index,
                "class_label": (
                    self.class_labels[index]
                    if index < len(self.class_labels)
                    else str(index)
                ),
                "probabilities": {
                    (
                        self.class_labels[i]
                        if i < len(self.class_labels)
                        else str(i)
                    ): round(p, 4)
                    for i, p in enumerate(probabilities)
                },
            },
        }
