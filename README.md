# Heart Disease Prediction 🫀

A machine-learning **web application** that predicts the likelihood of heart
disease from 13 clinical measurements. It is built from a neural network
trained on the public
[UCI Cleveland Heart Disease dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data).

This project converts the original Jupyter notebook
(`Heart disease prediction/Heart Disease Prediction.ipynb`) into a clean,
runnable Python script and turns it into a deployable Flask website.

- 🧠 **Six models, one ensemble** – two Keras neural networks (a 5-class
  severity model and a binary *disease / no-disease* model) **plus** Logistic
  Regression, Random Forest, Gaussian Naive Bayes and K-Nearest Neighbours.
  Every binary model votes and the final assessment is their ensemble average.
- 🌐 **Rich Flask web UI + JSON API** – a sticky **navigation menu**, grouped
  input sections, an animated **risk gauge**, **per-model prediction cards**, a
  **severity chart** and a **"key risk factors"** breakdown – all vanilla JS,
  no build step.
- 🔐 **Sign in / sign up + light mode** – the whole site (pages *and* API) sits
  behind a session login, with light-mode **sign-in** and **sign-up** pages so
  anyone can register an account. The interface opens in **light mode** with a
  light/dark toggle in the menu, and forms always start empty.
- ⚡ **Zero heavy dependencies at runtime** – the trained weights are exported
  to `model/weights.json` and evaluated in pure Python, so the deployed app
  needs **only Flask** (no TensorFlow, no scikit-learn). This is what keeps it
  inside Vercel's serverless function size limit.
- 🚀 **Ready for Vercel** – zero-configuration deployment.

---

## ✨ Features

| Feature | Description |
| --- | --- |
| Dataset | UCI Cleveland heart disease (297 patients after cleaning, 13 features) |
| Models | 5 binary classifiers + 1 five-class severity network → ensemble |
| Neural nets | `13 → 8 → 4 → 5` (softmax) and `13 → 8 → 4 → 1` (sigmoid) |
| Classic models | Logistic Regression, Random Forest, Gaussian NB, K-Nearest Neighbours |
| Backend | Flask (WSGI) |
| Frontend | Server-rendered HTML + vanilla JS: sticky menu, risk gauge, cards, charts |
| Auth | Session login **and self-service sign up** (`/login`, `/signup`); UI **and** API protected |
| Theme | **Light mode by default** + light/dark toggle in the top menu |
| Runtime deps | Flask only (`requirements.txt`) |
| Training deps | TensorFlow / Keras / scikit-learn (`requirements-train.txt`) |
| Hosting | Vercel (zero-config Flask preset) |

---

## 🧪 Models & accuracy

Every model is trained by `train.py` and evaluated on the **same held-out test
set** (60 patients, never seen during training). These are the numbers baked into
the shipped `model/weights.json` and displayed in the *Models* section of the web
page:

| Model | Task | Accuracy | ROC-AUC |
| --- | --- | --- | --- |
| Gaussian Naive Bayes | disease / no disease | **91.7%** | 0.946 |
| Logistic Regression | disease / no disease | **86.7%** | 0.942 |
| Random Forest | disease / no disease | **86.7%** | 0.940 |
| Neural Network | disease / no disease | 73.3% | 0.824 |
| K-Nearest Neighbours | disease / no disease | 65.0% | 0.670 |
| Neural Network | severity (5 classes) | 70.0% | — |
| **Ensemble** | disease / no disease | average of the 5 binary models | — |

Exact numbers vary slightly between runs/versions of the libraries — re-run
`python train.py` to regenerate them.

---

## 📁 Project structure

```
Heart-Disease-Prediction/
├── app.py                     # Flask web app (WSGI entrypoint: `app`)
├── model.py                   # Pure-Python inference (loads the weights)
├── train.py                   # Notebook ➜ runnable script (Keras training)
├── model/
│   ├── weights.json           # Trained weights + metadata (generated)
│   └── feature_histograms.png # EDA plot (generated)
├── templates/
│   ├── index.html             # Web UI
│   ├── login.html             # Sign-in page (light mode)
│   └── signup.html            # Sign-up page (light mode)
├── static/
│   ├── style.css              # Styles
│   └── app.js                 # Front-end logic
├── test_app.py                # Unit tests (unittest)
├── users.json                 # Registered accounts (generated, git-ignored)
├── requirements.txt           # Runtime dependencies (Flask) → used by Vercel
├── requirements-train.txt     # Training dependencies (TensorFlow, sklearn, …)
├── .python-version            # Pins Python 3.12 for Vercel
├── .gitignore
├── .vercelignore
└── Heart disease prediction/  # Original notebook & documents (archived)
```

---

## 🖥️ Web interface

The single page is organised into sections reachable from the **sticky top menu**:

| Menu item | Section | What it shows |
| --- | --- | --- |
| **Predict** | input form | 13 clinical fields grouped into *Demographics*, *Symptoms*, *Vitals & labs* and *ECG & tests* |
| **Models** | performance table | accuracy & ROC-AUC of every model |
| **Dataset** | reference table | feature ranges and means from the training data |
| **API** | docs | endpoints and a ready-to-copy `curl` example |
| **About** | info | how the project works + disclaimer |

After a prediction the results panel appears with:

- an **ensemble verdict** and an animated **risk gauge** (0–100%),
- **per-model cards** with each classifier's own probability and test accuracy,
- a **severity distribution** bar chart (the 5-class network),
- the **top risk factors**, with red/green bars for *increases* / *decreases* risk.

The page always opens in **light mode** — press the ☼ / ☾ button in the top menu
to switch themes (your choice is remembered by the browser). The input boxes are
always **empty on load**: nothing from a previous patient is kept, not even after
a refresh or a back/forward navigation.

---

## 🔐 Authentication

Everything except `/api/health` sits behind a session login, and the app supports
**both signing in and signing up**.

| Page | Purpose |
| --- | --- |
| `/login` | Sign in with an existing account (light mode) |
| `/signup` | Create a new account — signs you in straight away (light mode) |
| `/logout` | End the session |

* Passwords are hashed with `werkzeug.security` (`generate_password_hash` /
  `check_password_hash`) — they are never stored or logged in clear text.
* Registration rules: username **3–24** characters (`a–z A–Z 0–9 . _ -`),
  password **≥ 6** characters, the confirmation must match, and duplicate
  usernames are rejected.
* Visiting `/` or any `/api/*` endpoint without a session redirects browsers to
  `/login` and answers API clients with
  **`401 {"error": "Authentication required."}`**.
* The top menu shows the signed-in user and a **Sign out** link.
* Sessions are signed cookies valid for **7 days**.

### Where accounts are stored

Accounts created at `/signup` are written to **`users.json`** next to `app.py`
(git-ignored). The store tries, in order: `USERS_FILE` → `./users.json` → a
temporary file → memory, so the app keeps working even on a read-only
filesystem. Accounts defined through the environment are always present and take
precedence over a same-named account in the file.

### Default demo account

| Username | Password |
| --- | --- |
| `admin` | `heart123` |

### Environment variables

Set these locally, or in the Vercel dashboard under
*Settings → Environment Variables*:

| Variable | Purpose |
| --- | --- |
| `SECRET_KEY` | **Set this in production** — signs the session cookie |
| `APP_USERNAME` | Username for a single built-in account |
| `APP_PASSWORD` | Password for that account |
| `APP_USERS` | Several built-in accounts, e.g. `alice:secret,bob:hunter2` |
| `USERS_FILE` | Where registered accounts are stored (default `users.json`) |

```powershell
$env:SECRET_KEY="a-long-random-string"
$env:APP_USERNAME="ramya"; $env:APP_PASSWORD="my-password"
python app.py
```

> The sign-in / sign-up pages show the demo credentials as a hint. Delete the
> `.auth-hint` block from `templates/login.html` and `templates/signup.html` if
> you don't want that.

> ⚠️ **Serverless hosts (Vercel):** the deployment bundle is read-only, so
> `users.json` lands in a temporary directory — registrations do **not** survive
> a redeploy and are not shared between instances. For real multi-user
> sign-up on Vercel, point `USERS_FILE` at persistent storage or swap the store
> for a database (e.g. Vercel Postgres / KV).

---

## 🚀 Run locally

> Requires **Python 3.10+**. The steps below work on Windows, macOS and Linux.

### 1. Clone the repository

```bash
git clone https://github.com/17ramya/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the runtime dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the web app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser. You are redirected to the
**sign-in page** first — use `admin` / `heart123` (or your own `APP_USERNAME` /
`APP_PASSWORD`). After signing in, fill in the patient data and click
**Predict**.

> The repository already ships a trained `model/weights.json`, so you can run
> the website **without** installing TensorFlow.

---

## 🔁 Retrain the models (optional)

The training script needs TensorFlow and friends. Install them **into the same
virtual environment**:

```bash
pip install -r requirements-train.txt
```

Then run:

```bash
python train.py
```

This will:

1. Download the Cleveland dataset (or use `data/processed.cleveland.data` if
   you place a local copy there).
2. Clean and split the data.
3. Train **both neural networks** (100 epochs each) **and** the four classic
   models (Logistic Regression, Random Forest, Gaussian Naive Bayes,
   K-Nearest Neighbours).
4. Print the accuracy / ROC-AUC comparison for every model.
5. **Overwrite `model/weights.json`** with all of the trained weights plus
   metadata, and save a histogram plot to `model/feature_histograms.png`.

Restart `python app.py` afterwards to serve the freshly trained model.

> `train.py` fixes the train/test split with `random_state=42` (the notebook did
> not) so that re-training is reproducible.

---

## 🔌 API reference

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/login` | Sign-in page (light mode) |
| `POST` | `/login` | Submit credentials (`username`, `password`, optional `next`) |
| `GET` | `/signup` | Sign-up page (light mode) |
| `POST` | `/signup` | Register (`username`, `password`, `confirm`) → auto sign-in |
| `GET` | `/logout` | End the session |
| `GET` | `/` | Web UI *(requires login)* |
| `GET` | `/api/health` | Health check → `{"status": "ok"}` *(public)* |
| `GET` | `/api/session` | Current session → `{"authenticated": true, "user": "admin"}` |
| `GET` | `/api/metadata` | Feature definitions + model metadata *(requires login)* |
| `POST` | `/api/predict` | Run a prediction *(requires login)* |

> Every protected endpoint redirects browsers to `/login` (302) but answers
> `/api/*` requests with **`401 {"error": "Authentication required."}`**, so you
> must send the session cookie when calling the API from a script.

**Example request**

```bash
# 1. sign in once and keep the session cookie
curl -c jar.txt -X POST http://127.0.0.1:5000/login \
  -d "username=admin&password=heart123"

# 2. call the protected endpoint with that cookie
curl -b jar.txt -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,
       "restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,
       "ca":0,"thal":6}'
```

**Example response**

```json
{
  "binary": {
    "prediction": "No Heart Disease",
    "label": 0,
    "probability": 0.3421,
    "confidence": 0.6579
  },
  "models": [
    { "key": "nn_binary",     "label": "Neural Network",        "probability": 0.069, "prediction": "No Heart Disease", "accuracy": 0.7333, "roc_auc": 0.8241 },
    { "key": "logistic",      "label": "Logistic Regression",   "probability": 0.0,   "prediction": "No Heart Disease", "accuracy": 0.8667, "roc_auc": 0.9421 },
    { "key": "random_forest", "label": "Random Forest",         "probability": 0.409, "prediction": "No Heart Disease", "accuracy": 0.8667, "roc_auc": 0.9398 },
    { "key": "gaussian_nb",   "label": "Gaussian Naive Bayes",  "probability": 0.0,   "prediction": "No Heart Disease", "accuracy": 0.9167, "roc_auc": 0.9456 },
    { "key": "knn",           "label": "K-Nearest Neighbours",  "probability": 0.429, "prediction": "No Heart Disease", "accuracy": 0.65,   "roc_auc": 0.6701 }
  ],
  "categorical": {
    "class": 0,
    "class_label": "No disease (0)",
    "probabilities": { "No disease (0)": 0.41, "Mild (1)": 0.22, "Moderate (2)": 0.18, "Severe (3)": 0.12, "Very severe (4)": 0.07 }
  },
  "risk_factors": [
    { "feature": "fbs", "contribution": -1.0087, "increases_risk": false, "raw_value": 1.0 }
  ]
}
```

> `binary` is the **ensemble** (mean of the five binary models); `models` lists
> each model's own verdict; `risk_factors` are the top logistic-regression
> contributions (`increases_risk: true` = pushes the risk *up*).

---

## ✅ Run the tests

```bash
pip install -r requirements.txt
python -m unittest test_app -v
```

---

## ☁️ Deploy to Vercel

The project uses Vercel's **zero-configuration Flask** support: Vercel detects
the `Flask` dependency in `requirements.txt` and loads the `app` object from
`app.py`. **No `vercel.json` is required.**

Because the app performs inference in pure Python (`model.py`), it does **not**
bundle TensorFlow, keeping the serverless function well under Vercel's size
limit.

### Option A — Deploy from GitHub (recommended)

1. Push this repository to GitHub (already done if you cloned it).
2. Go to <https://vercel.com/new> and **Import** the repository.
3. Vercel auto-detects the framework. Leave the defaults:
   - **Framework Preset:** `Flask`
   - **Build Command:** *(leave empty)*
   - **Output Directory:** *(leave empty)*
4. *(Recommended)* add your credentials under **Settings → Environment
   Variables**: `SECRET_KEY` (a long random string), `APP_USERNAME`,
   `APP_PASSWORD` — see [Authentication](#-authentication). Without a fixed
   `SECRET_KEY` the app falls back to a development default.
5. Click **Deploy**. Your app is live at
   `https://<project-name>.vercel.app`.

### Option B — Deploy with the Vercel CLI

```bash
npm install -g vercel    # one-time
vercel login             # one-time
vercel                   # preview deployment
vercel --prod            # production deployment
```

### Python version

`.python-version` pins the deployment to **Python 3.12**. To change it, edit
that file (e.g. `3.13`).

---

## 🛠️ Troubleshooting

| Problem | Fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'flask'` | Activate the virtual environment, then `pip install -r requirements.txt`. |
| `python app.py` runs but the page shows a 500 error | Make sure `model/weights.json` exists (run `python train.py` to regenerate it). |
| `train.py` fails to download the dataset | Place a local copy at `data/processed.cleveland.data` and re-run. |
| Vercel build tries to install TensorFlow | Confirm `requirements.txt` contains only `Flask`; training deps live in `requirements-train.txt`. |
| Port 5000 already in use | Run with `set PORT=5001 && python app.py` (Windows) or `PORT=5001 python app.py` (macOS/Linux). |
| Can't sign in | Use `admin` / `heart123`, or the value of `APP_USERNAME` / `APP_PASSWORD` if you set them. |
| "That username is already taken" | Pick another username, or use **Sign in** instead of **Create one**. |
| "Password must be at least 6 characters" | Registration requires ≥ 6 characters (see `MIN_PASSWORD_LENGTH` in `app.py`). |
| Registered users vanish after a redeploy (Vercel) | Expected — the bundle is read-only so `users.json` falls back to `/tmp`. Point `USERS_FILE` at persistent storage or use a database. |
| Signed out after every deploy (Vercel) | Set a fixed `SECRET_KEY` environment variable so the session cookie is always signed with the same key. |
| `401 {"error":"Authentication required."}` from a script | Sign in first and reuse the cookie: `curl -c jar.txt -d "username=admin&password=heart123" …/login`, then add `-b jar.txt`. |
| An input box shows an old value | Hard-refresh (Ctrl+F5). The page clears every field on load and sets `autocomplete="off"` on all inputs. |

---

## 📄 Notes & disclaimer

- The original notebook and its exported `.py` file are kept under
  `Heart disease prediction/` for reference. `train.py` **supersedes** that
  broken export (the old file contained bare text like `IMPORTING DATASET:`
  which is invalid Python).
- Feature values are used **raw**, exactly as in the notebook (no
  normalisation), so the web form expects the original UCI encodings. Logistic
  Regression standardises internally (the mean/scale it learned is stored in the
  weights file).
- Input values are always assembled in the dataset's **column order** before
  being passed to the models, so changing how the form groups or orders fields
  never affects predictions.
- Login state lives in a signed session cookie (set `SECRET_KEY`). Passwords are
  stored as salted hashes via `werkzeug.security`; the built-in demo account is
  `admin` / `heart123`, new accounts can be created at `/signup` (kept in
  `users.json`), and every page except `/api/health` requires a session.
- **This project is an educational machine-learning demo on a small public
  dataset. It is not a medical device and must not be used for real clinical
  decisions.**

