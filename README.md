# Exoplanet Habitability Predictor

A lightweight Python project that uses a neural network to predict the probability that an exoplanet could support life, based on NASA’s Exoplanet Archive. Includes:

- **Data preprocessing** and cleaning  
- **Baseline** logistic‐regression model  
- **Feed-forward PyTorch neural network**  
- **Threshold tuning** to balance precision vs. recall  
- **Interactive demo** via Jupyter+ipywidgets  

---

## Core Features

Our model relies on eight physically meaningful inputs, each directly tied to habitability criteria:

| Feature       | Units                  | Meaning                                                                 |
|--------------:|-----------------------:|-------------------------------------------------------------------------|
| **pl_rade**   | Earth radii            | Planet radius, distinguishing rocky worlds (smaller) from gas giants   |
| **pl_bmasse** | Earth masses           | Planet mass, combined with radius to infer bulk density                 |
| **pl_orbsmax**| Astronomical units     | Semi-major axis: distance from host star, controlling temperature       |
| **pl_orbeccen**| (unitless)            | Orbital eccentricity: high values cause extreme temperature swings      |
| **pl_insol**  | Earth flux             | Insolation: stellar energy received, key for being in the “Goldilocks” zone |
| **st_teff**   | Kelvin                 | Stellar effective temperature, determining star’s spectral output       |
| **st_rad**    | Solar radii            | Stellar radius, affecting the size and distance of the habitable zone   |
| **st_mass**   | Solar masses           | Stellar mass, correlating with luminosity and main-sequence lifetime    |

## 📁 Project Structure

```
exoplanet-habitability-predictor/
├── data/
│   ├── exoplanets_raw.csv
│   ├── exoplanets_processed.csv
│   └── artifacts/
│       ├── X_train.npy
│       ├── y_train.npy
│       ├── X_val.npy
│       ├── y_val.npy
│       ├── scaler.joblib
│       ├── model.pth
│       ├── model_probs.npy
│       └── config.json
├── notebooks/
│   ├── 01_explore_data.ipynb
│   ├── 02_train_and_eval.ipynb
│   ├── 03_threshold_tuning.ipynb
│   └── 04_user_prediction.ipynb
├── src/
│   ├── preprocess.py
│   ├── model.py
│   └── train.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/exoplanet-habitability-predictor.git
   cd exoplanet-habitability-predictor
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   # macOS/Linux
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## ⚙️ Usage

### 1. Run the Notebooks

All data prep, training, evaluation, and demos live in `notebooks/`. You can launch them:

- **Locally:**
  ```bash
  jupyter lab
  ```

- **In the cloud using Binder (no install needed):**

  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/<your-username>/exoplanet-habitability-predictor/HEAD)

### 2. Preprocess & Train via Scripts

If you prefer CLI scripts:

```bash
python src/preprocess.py    # Cleans raw CSV and saves processed data
python src/train.py         # Trains baseline + NN, saves artifacts & probs
```

### 3. Interactive Demo

Open `notebooks/04_user_prediction.ipynb` and run the cell to get a form where you can enter the 8 model inputs and click Predict.

---

## 📈 Results

**Baseline logistic regression:**
- Accuracy ≈ 97.5%
- ROC-AUC ≈ 0.865

**Neural network after training:**
- Accuracy ≈ 97.8%
- ROC-AUC ≈ 0.981

**Threshold tuning** (F1-maximizing cutoff ≈ 0.173) yields:
- Accuracy ≈ 98.2%
- Precision ≈ 55.6%
- Recall ≈ 88.2%

---

## 🤝 Contributing

Feel free to open issues or pull requests. Possible extensions:

- Add feature engineering (bulk density, luminosity)
- Interpretability with SHAP/LIME
- Deploy as a lightweight web service

---

## 📄 License

This project is released under the MIT License.

```
MIT License

Copyright (c) 2025 <your-name>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```