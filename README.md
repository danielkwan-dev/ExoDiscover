# ExoDiscover

Exoplanet classification system using machine learning on NASA Kepler mission data.

## Demo

### Option 1: Run the Website Locally

This starts the Flask backend + the Vite frontend.

**Backend setup:**
```bash
cd exodiscover-backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Start the API:**
```bash
python app.py
```

**Start the frontend (in a second terminal):**
```bash
npm install
npm run dev
```

Open the app using the given localhost server.

### Option 2: Run Predictions via Script

This runs a script-driven predictor and does not use the web app.

Ensure the Python environment is ready (same as Option 1), then run:
```bash
python exodiscover-backend/predict.py
```

## ML Component

**Data source:** NASA Kepler Objects of Interest (KOI) catalog

**Features extracted from light curves:**
- Statistical: mean, std, median, max, min, range, coefficient of variation
- Transit detection: num_dips, dip_fraction, max_dip_depth, transit_count
- Signal analysis: skewness, kurtosis, SNR, smoothness
- Periodicity: num_periods, primary_period

**Target labels:**
- Confirmed (verified exoplanet)
- Candidate (potential exoplanet)
- False Positive (not an exoplanet)

**Models implemented using scikit-learn and XGBoost:**
- XGBoost Classifier (production)
- CNN for light curve analysis
- LSTM for sequential data

**Training and evaluation:**
```bash
python exodiscover-backend/train_final_model.py
python exodiscover-backend/predict.py
```

**Evaluation metrics:**
- Accuracy on holdout test set
- Classification report (precision, recall, F1)
- Confusion matrix

## Tech Stack

**Frontend:**
- React
- TypeScript
- Vite
- Tailwind CSS
- shadcn-ui

**Backend:**
- Flask
- XGBoost
- scikit-learn
- TensorFlow/Keras
- lightkurve (NASA light curve analysis)

## Data Sources

- NASA Exoplanet Archive
- Kepler Objects of Interest (KOI) catalog
- Kepler, K2, and TESS mission data
