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
│   └── index.html             # Web UI
├── static/
│   ├── style.css              # Styles
│   └── app.js                 # Front-end logic
├── test_app.py                # Unit tests (unittest)
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

Open **http://127.0.0.1:5000** in your browser, fill in the patient data and
click **Predict**.

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
| `GET` | `/` | Web UI |
| `GET` | `/api/health` | Health check → `{"status": "ok"}` |
| `GET` | `/api/metadata` | Feature definitions + model metadata |
| `POST` | `/api/predict` | Run a prediction |

**Example request**

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
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
4. Click **Deploy**. Your app is live at
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
- **This project is an educational machine-learning demo on a small public
  dataset. It is not a medical device and must not be used for real clinical
  decisions.**

