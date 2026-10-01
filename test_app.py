"""Tests for the Heart Disease Prediction web app.

Run with::

    python -m unittest test_app -v
"""

import json
import os
import re
import shutil
import tempfile
import unittest

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
        # A throwaway user database, so tests never touch the real users.json
        # and never depend on the environment accounts.
        self.users_dir = tempfile.mkdtemp()
        store = app_module.UserStore(os.path.join(self.users_dir, "users.json"))
        store.add(TEST_USER, TEST_PASSWORD)
        app_module.USERS = store
        self.client = app.test_client()

    def tearDown(self):
        shutil.rmtree(self.users_dir, ignore_errors=True)

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
    # registration (sign up)
    # ------------------------------------------------------------------ #

    def signup(self, username, password, confirm=None):
        return self.client.post(
            "/signup",
            data={
                "username": username,
                "password": password,
                "confirm": password if confirm is None else confirm,
            },
            follow_redirects=False,
        )

    def test_signup_page_is_light_mode(self):
        resp = self.client.get("/signup")
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode()
        self.assertIn("Create account", html)
        self.assertIn('data-theme="light"', html)

    def test_login_page_links_to_signup(self):
        html = self.client.get("/login").data.decode()
        self.assertIn('href="/signup"', html)

    def test_signup_creates_account_and_signs_in(self):
        resp = self.signup("newcomer", "hunter2!")
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(self.client.get("/").status_code, 200)  # auto signed in

        # ... and the new account can sign in again afterwards
        self.client.get("/logout")
        self.assertEqual(self.client.get("/").status_code, 302)
        self.assertEqual(self.login("newcomer", "hunter2!").status_code, 302)

    def test_signup_account_reaches_the_api(self):
        self.signup("apiuser", "hunter2!")
        resp = self.client.post("/api/predict", json=VALID_PAYLOAD)
        self.assertEqual(resp.status_code, 200)

    def test_signup_persists_to_disk(self):
        self.signup("diskuser", "hunter2!")
        with open(app_module.USERS.path, encoding="utf-8") as handle:
            saved = json.load(handle)
        self.assertIn("diskuser", saved)
        self.assertIn("password", saved["diskuser"])
        self.assertNotIn("hunter2!", json.dumps(saved))  # never stored in clear

    def test_signup_rejects_duplicate_username(self):
        resp = self.signup(TEST_USER, "another-pass")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("already taken", resp.data.decode())

    def test_signup_rejects_short_password(self):
        resp = self.signup("shorty", "123")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("at least", resp.data.decode())

    def test_signup_rejects_mismatched_confirmation(self):
        resp = self.signup("mismatch", "hunter2!", confirm="something-else")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("do not match", resp.data.decode())

    def test_signup_rejects_invalid_username(self):
        resp = self.signup("no spaces allowed", "hunter2!")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Username must be", resp.data.decode())

    def test_signup_unavailable_when_logged_in(self):
        self.login()
        self.assertEqual(self.client.get("/signup").status_code, 302)

    def test_default_account_always_present(self):
        # The environment/default account is seeded into the store.
        self.assertTrue(app_module.authenticate(TEST_USER, TEST_PASSWORD))
        self.assertIsNone(app_module.authenticate(TEST_USER, "wrong"))

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

