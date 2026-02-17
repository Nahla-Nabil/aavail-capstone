# AAVAIL Capstone Project — AI Workflow: AI in Production

## Project Structure

```
capstone/
├── app.py                  ← Flask API (train, predict, logs endpoints)
├── run_tests.py            ← Run ALL tests with one command
├── requirements.txt        ← Python packages
├── Dockerfile              ← Docker container
├── eda.ipynb               ← EDA + model comparison notebook
├── src/
│   ├── ingest_data.py      ← Data loading & feature engineering
│   ├── model.py            ← Model training, comparison, prediction
│   └── logger.py           ← Logging & performance monitoring
├── tests/
│   ├── test_api.py         ← API unit tests
│   ├── test_model.py       ← Model unit tests
│   └── test_logger.py      ← Logger unit tests
├── models/                 ← Saved model files (.pkl)
├── logs/                   ← JSON log files
└── data/                   ← Your JSON data files go here
```

---

## Step-by-Step Setup

### Step 1: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 2: Add Your Data
Place the AAVAIL JSON files into the `data/` folder.

### Step 3: Run EDA Notebook
```bash
jupyter notebook eda.ipynb
```

### Step 4: Train the Model via API
Start the API first:
```bash
python app.py
```

Then train:
```bash
# Train for all countries
curl -X POST http://localhost:8080/train -H "Content-Type: application/json" -d '{"country": "all"}'

# Train for a specific country
curl -X POST http://localhost:8080/train -H "Content-Type: application/json" -d '{"country": "united_kingdom"}'
```

### Step 5: Get Predictions
```bash
# Predict for all countries
curl http://localhost:8080/predict?country=all

# Predict for specific country
curl http://localhost:8080/predict?country=united_kingdom

# Predict for specific date
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" \
  -d '{"country": "germany", "date": "2019-03-15"}'
```

### Step 6: Run All Tests
```bash
python run_tests.py
```

### Step 7: Build & Run with Docker
```bash
# Build image
docker build -t aavail-capstone .

# Run container
docker run -p 8080:8080 aavail-capstone

# Test it
curl http://localhost:8080/health
```

---

## Peer Review Checklist

| Question | Answer |
|---|---|
| Unit tests for the API? | ✅ `tests/test_api.py` |
| Unit tests for the model? | ✅ `tests/test_model.py` |
| Unit tests for the logging? | ✅ `tests/test_logger.py` |
| Single script for all tests? | ✅ `python run_tests.py` |
| Mechanism to monitor performance? | ✅ `src/logger.py` — Wasserstein distance + MAE/RMSE tracking |
| Tests isolated from production? | ✅ Uses temp directories and mock objects |
| API works for specific country AND all countries? | ✅ `?country=uk` or `?country=all` |
| Data ingestion as function/script? | ✅ `src/ingest_data.py` — `fetch_data()` function |
| Multiple models compared? | ✅ LinearRegression, RandomForest, GradientBoosting |
| EDA uses visualizations? | ✅ `eda.ipynb` — 4+ charts |
| Containerized with Docker? | ✅ `Dockerfile` |
| Visualization comparing to baseline? | ✅ `predictions_vs_baseline.png` in notebook |
