"""Tests for the Heart Disease Prediction web app.

Run with::

    python -m unittest test_app -v
"""

import re
import unittest

from werkzeug.security import generate_password_hash

import app as app_module
from app import FEATURES, app


VALID_PAYLOAD = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}

TEST_USER = "tester"
TEST_PASSWORD = "s3cret-pass"


class HeartDiseaseAppTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        # Deterministic user store that does not depend on the environment.
        app_module.USERS = {TEST_USER: generate_password_hash(TEST_PASSWORD)}
        self.client = app.test_client()

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def login(self, username=TEST_USER, password=TEST_PASSWORD):
        return self.client.post(
            "/login",
            data={"username": username, "password": password},
            follow_redirects=False,
        )

    # ------------------------------------------------------------------ #
    # authentication
    # ------------------------------------------------------------------ #

    def test_login_page_is_light_mode(self):
        resp = self.client.get("/login")
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode()
        self.assertIn("Sign in", html)
        self.assertIn('data-theme="light"', html)

    def test_index_requires_login(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login", resp.headers["Location"])

    def test_api_requires_login(self):
        resp = self.client.post("/api/predict", json=VALID_PAYLOAD)
        self.assertEqual(resp.status_code, 401)
        self.assertIn("error", resp.get_json())

    def test_metadata_requires_login(self):
        self.assertEqual(self.client.get("/api/metadata").status_code, 401)

    def test_login_success_then_logout(self):
        resp = self.login()
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/", resp.headers["Location"])
        self.assertEqual(self.client.get("/").status_code, 200)

        self.client.get("/logout")
        self.assertEqual(self.client.get("/").status_code, 302)

    def test_login_failure(self):
        resp = self.login(password="not-the-password")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Incorrect username or password", resp.data.decode())

    def test_session_endpoint(self):
        self.assertFalse(self.client.get("/api/session").get_json()["authenticated"])
        self.login()
        self.assertTrue(self.client.get("/api/session").get_json()["authenticated"])

    def test_health_is_public(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["status"], "ok")

    # ------------------------------------------------------------------ #
    # application
    # ------------------------------------------------------------------ #

    def test_feature_count(self):
        self.assertEqual(len(FEATURES), 13)

    def test_index_page(self):
        self.login()
        html = self.client.get("/").data.decode()
        self.assertIn("Heart Disease Prediction", html)
        self.assertIn("Model performance", html)
        self.assertIn("nav-links", html)      # navigation menu
        self.assertIn("theme-toggle", html)   # light / dark switch
        self.assertIn(TEST_USER, html)        # signed-in user shown
        self.assertIn("Sign out", html)

    def test_form_starts_empty(self):
        self.login()
        html = self.client.get("/").data.decode()
        self.assertIn("Select&hellip;", html)          # empty select placeholder
        self.assertIn('placeholder="e.g.', html)       # hint instead of a value

        age = re.search(r'<input[^>]*name="age"[^>]*>', html)
        self.assertIsNotNone(age)
        self.assertNotIn("value=", age.group(0))       # nothing pre-filled

    def test_logged_in_user_cannot_see_login_form(self):
        self.login()
        resp = self.client.get("/login")
        self.assertEqual(resp.status_code, 302)

    def test_metadata(self):
        self.login()
        body = self.client.get("/api/metadata").get_json()
        self.assertEqual(len(body["features"]), 13)
        self.assertIn("stats", body["features"][0])
        self.assertIn("placeholder", body["features"][0])
        self.assertIn("models", body["model"])

    def test_predict_valid(self):
        self.login()
        resp = self.client.post("/api/predict", json=VALID_PAYLOAD)
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()

        # ensemble verdict
        self.assertIn(body["binary"]["prediction"],
                      ["Heart Disease", "No Heart Disease"])
        self.assertGreaterEqual(body["binary"]["probability"], 0.0)
        self.assertLessEqual(body["binary"]["probability"], 1.0)

        # every model reported
        self.assertEqual(len(body["models"]), 5)
        for model in body["models"]:
            self.assertGreaterEqual(model["probability"], 0.0)
            self.assertLessEqual(model["probability"], 1.0)

        # severity + risk factors
        self.assertEqual(len(body["categorical"]["probabilities"]), 5)
        self.assertTrue(len(body["risk_factors"]) >= 1)

        # regression test: the value reported for each factor must match the
        # input actually supplied for that feature (guards column ordering).
        for factor in body["risk_factors"]:
            self.assertEqual(
                factor["raw_value"], float(VALID_PAYLOAD[factor["feature"]])
            )

    def test_predict_missing_field(self):
        self.login()
        payload = dict(VALID_PAYLOAD)
        del payload["chol"]
        resp = self.client.post("/api/predict", json=payload)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.get_json())

    def test_predict_bad_value(self):
        self.login()
        payload = dict(VALID_PAYLOAD)
        payload["age"] = "not-a-number"
        resp = self.client.post("/api/predict", json=payload)
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)

