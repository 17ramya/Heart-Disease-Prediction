"""Basic tests for the Heart Disease Prediction web app.

Run with::

    python -m unittest test_app -v
"""

import unittest

from app import app


VALID_PAYLOAD = {
    "age": 63,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": 6,
}


class HeartDiseaseAppTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_index_page(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Heart Disease Prediction", resp.data)

    def test_health(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["status"], "ok")

    def test_metadata(self):
        resp = self.client.get("/api/metadata")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(len(body["features"]), 13)

    def test_predict_valid(self):
        resp = self.client.post("/api/predict", json=VALID_PAYLOAD)
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIn(body["binary"]["prediction"], ["Heart Disease", "No Heart Disease"])
        self.assertGreaterEqual(body["binary"]["probability"], 0.0)
        self.assertLessEqual(body["binary"]["probability"], 1.0)
        self.assertEqual(len(body["categorical"]["probabilities"]), 5)

    def test_predict_missing_field(self):
        payload = dict(VALID_PAYLOAD)
        del payload["chol"]
        resp = self.client.post("/api/predict", json=payload)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.get_json())

    def test_predict_bad_value(self):
        payload = dict(VALID_PAYLOAD)
        payload["age"] = "not-a-number"
        resp = self.client.post("/api/predict", json=payload)
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
