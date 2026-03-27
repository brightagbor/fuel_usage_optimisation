# AutoMPG · Fuel Intelligence

**A Streamlit application for predicting vehicle fuel efficiency and finding the optimal car configuration using Bayesian optimisation.**



## Overview

AutoMPG takes a set of car specifications as input, predicts the expected fuel efficiency in miles per gallon (MPG) using a trained ensemble ML model, and simultaneously runs a Gaussian Process optimisation to find the configuration within your defined search bounds that maximises MPG.

Both the prediction and the optimisation run with a single button click — no separate steps required.



## Features

- **MPG Prediction** — Instantly predicts fuel efficiency for any car configuration you specify
- **Bayesian Optimisation** — Uses `gp_minimize` (Gaussian Process with Expected Improvement) to search for the globally optimal car configuration within user-defined bounds
- **Feature Comparison Table** — Side-by-side view of your input values vs the optimal values, with directional change indicators and inline bar charts
- **Auto Scaler Detection** — Automatically detects and applies a companion scaler/preprocessor if the model was trained on normalised data
- **Live Progress Bar** — Real-time callback shows optimisation progress and best MPG found so far
- **Sidebar Diagnostics** — On startup, probes the model with known extreme inputs to verify it is responding correctly to feature changes



## Project Structure

```
project/
├── app.py                        # Main Streamlit application
├── README.md                     # This file
└── results/
    ├── ensemble1_best_model.pkl  # Trained ensemble model (required)
    └── scaler.pkl                # Feature scaler (required if model was trained on scaled data)
```

The app will automatically look for a scaler under any of these names inside `results/`:

```
scaler.pkl
ensemble1_scaler.pkl
preprocessor.pkl
ensemble1_preprocessor.pkl
```

If your model is a scikit-learn `Pipeline` that already includes preprocessing steps, no separate scaler file is needed.



## Input Features

| Feature | Type | Range | Description |
|||||
| Cylinders | Integer | 3 – 8 | Number of engine cylinders |
| Displacement | Float | 50 – 500 cu·in | Engine displacement |
| Horsepower | Float | 40 – 250 hp | Engine horsepower |
| Weight | Float | 1500 – 5500 lbs | Vehicle weight |
| Acceleration | Float | 8 – 25 s | Time to accelerate 0–60 mph |
| Model Year | Integer | 1970 – 1982 | Model year (encoded as 70–82) |
| Origin | Integer | 1, 2, 3 | Manufacturing origin (1=USA, 2=Europe, 3=Japan) |



## Optimisation Details

| Setting | Value |
|||
| Algorithm | Gaussian Process (`gp_minimize`) |
| Acquisition function | Expected Improvement (EI) |
| Total evaluations | 80 |
| Initial random points | 20 |
| Noise assumption | 1e-10 (deterministic model) |
| Exploration factor (ξ) | 0.05 |

The optimisation searches within the bounds you set using the sliders in the **Optimisation Search Bounds** panel. Narrowing these bounds focuses the search on realistic or constrained configurations. Wider bounds allow more global exploration.

Integer features (Cylinders, Model Year, Origin) are rounded after each evaluation and at the final step to ensure the reported optimal configuration is always a valid discrete value.



## Installation

**1. Clone or copy the project files into a folder.**

**2. Install dependencies:**

```bash
pip install streamlit numpy scikit-optimize joblib scikit-learn
```

**3. Place your trained model file at:**

```
results/ensemble1_best_model.pkl
```

And your scaler (if applicable) at:

```
results/scaler.pkl
```

**4. Run the app:**

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.



## Usage

1. **Enter your car's specifications** in the Car Specifications section — cylinders, displacement, horsepower, weight, acceleration, model year, and origin.
2. **Adjust the optimisation bounds** using the sliders if you want to restrict the search to a specific design space (e.g. only 4-cylinder engines, or only Japanese-origin cars).
3. **Click the button** — *Predict MPG & Find Optimal Configuration*.
4. The app will:
   - Immediately predict the MPG for your entered configuration
   - Run 80 Bayesian optimisation evaluations across the bounds you set
   - Display both values side by side with the percentage improvement
   - Show a full feature comparison table with change indicators



## Troubleshooting

**Both MPG values are identical**

Open the sidebar panel. If it says *"Model outputs near-identical values for very different inputs"*, the model requires a scaler that isn't present. Save your training scaler to `results/scaler.pkl` and restart the app.

**`FileNotFoundError` on startup**

The model file is missing. Ensure `results/ensemble1_best_model.pkl` exists relative to where you run `streamlit run`.

**Optimisation returns values inside the bounds but MPG doesn't improve**

Try widening the optimisation bounds — if the bounds are very narrow, the GP has little room to find a better configuration than your input. Also ensure Origin and Model Year bounds span more than a single value.

**`ModuleNotFoundError`**

Install any missing package with `pip install <package-name>`. The full list of required packages is in the Installation section above.



## Dataset

The model was trained on the **UCI Auto-MPG dataset**, which contains fuel consumption data for cars manufactured between 1970 and 1982. The dataset covers American, European, and Japanese vehicles.



## Model

The prediction model is an **Ensemble Regressor** (`ensemble1_best_model.pkl`). It combines multiple base estimators to produce a robust MPG estimate. The exact ensemble composition depends on your training script.



## Dependencies

| Package | Purpose |
|||
| `streamlit` | Web application framework |
| `numpy` | Numerical array operations |
| `scikit-optimize` | Bayesian optimisation (`gp_minimize`) |
| `joblib` | Model serialisation / loading |
| `scikit-learn` | Scaler detection and Pipeline support |