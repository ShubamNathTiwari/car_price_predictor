# 🚗 AutoValue AI — Car Price Prediction System

> A full-fledged Machine Learning project with a stunning 3D-styled UI, AI-powered price predictions, and head-to-head car comparison.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔮 **Price Predictor** | Predict resale value using 4 trained ML models |
| ⚡ **Car Comparison** | Side-by-side comparison with radar charts & bar graphs |
| 📊 **Analytics Hub** | 3D scatter plots, heatmaps, violin charts, surface plots |
| 🤖 **Model Intelligence** | Performance leaderboard, feature importance, AI insights |
| 🎨 **3D UI** | Futuristic dark theme with glassmorphism & neon glows |

---

## 🚀 How to Run (Step-by-Step)

### Prerequisites
- Python 3.9 or higher — download from [python.org](https://www.python.org/downloads/)
- pip (comes with Python)

---

### ▶️ Option 1: One-Click Setup (Recommended)

**Linux / macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```
Double-click setup.bat
```

This automatically: creates a virtual environment → installs packages → trains models → launches the app.

---

### ▶️ Option 2: Manual Setup

**Step 1 — Create a virtual environment:**
```bash
python -m venv venv
```

**Step 2 — Activate it:**
```bash
# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**Step 3 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 4 — (Optional) Use your own dataset:**
Replace `car_data.csv` with your file. It must have these columns:
```
Car_Name, Year, Selling_Price, Present_Price, Kms_Driven,
Fuel_Type, Seller_Type, Transmission, Owner
```

**Step 5 — Train the ML models:**
```bash
python model_trainer.py
```
You will see training output like:
```
✅ Random Forest: R² = 96.12%, MAE = 0.4231
✅ Gradient Boosting: R² = 95.88%, MAE = 0.4512
🏆 Best Model: Random Forest (R² = 96.12%)
📦 All artifacts saved to model_artifacts/
```

**Step 6 — Launch the app:**
```bash
streamlit run app.py
```

**Step 7 — Open in browser:**
```
http://localhost:8501
```

---

## 🗂️ Project Structure

```
car_price_predictor/
│
├── app.py                  ← Main Streamlit application (UI + logic)
├── model_trainer.py        ← ML model training script
├── car_data.csv            ← Dataset (replace with your own)
├── requirements.txt        ← All Python dependencies
├── setup.sh                ← One-click setup for Linux/macOS
├── setup.bat               ← One-click setup for Windows
│
└── model_artifacts/        ← Auto-created after training
    ├── random_forest.pkl
    ├── gradient_boosting.pkl
    ├── ridge_regression.pkl
    ├── xgboost.pkl          ← (if XGBoost installed)
    ├── label_encoders.pkl
    ├── feature_cols.pkl
    ├── model_results.pkl
    └── dataset_stats.pkl
```

---

## 🤖 ML Models Used

| Model | Strength |
|---|---|
| **Random Forest** | Best overall accuracy, handles non-linearity |
| **Gradient Boosting** | Sequential ensemble, excellent performance |
| **Ridge Regression** | Fast baseline, handles multicollinearity |
| **XGBoost** | Extreme gradient boosting (optional) |

---

## 🎨 UI Pages

1. **🏠 Dashboard** — Live metrics, violin plots, 3D scatter, donut charts
2. **🔮 Price Predictor** — Input form + animated price display + depreciation gauge
3. **⚡ Car Comparison** — Dual-panel with floating car emojis, radar chart, bar comparison
4. **📊 Analytics Hub** — Trend lines, heatmaps, 3D surface plots, top-10 charts
5. **🤖 Model Intelligence** — Leaderboard, performance radar, feature importance

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `model_artifacts not found` | Run `python model_trainer.py` first |
| `Port already in use` | Run `streamlit run app.py --server.port 8502` |
| `XGBoost install fails` | Remove xgboost from requirements.txt (optional) |
| White screen on load | Refresh browser / clear cache |
| Streamlit not found | Ensure venv is activated before running |

---

## 📊 Dataset Columns Reference

| Column | Type | Description |
|---|---|---|
| `Car_Name` | str | Car model name |
| `Year` | int | Manufacturing year |
| `Selling_Price` | float | Resale price (₹ Lakhs) — **target** |
| `Present_Price` | float | Current ex-showroom price (₹ Lakhs) |
| `Kms_Driven` | int | Total kilometers driven |
| `Fuel_Type` | str | Petrol / Diesel / CNG |
| `Seller_Type` | str | Dealer / Individual |
| `Transmission` | str | Manual / Automatic |
| `Owner` | int | 0=1st, 1=2nd, 2=3rd owner |

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit + Custom CSS (glassmorphism, neon, 3D effects)
- **Visualization:** Plotly (3D scatter, surface, radar, gauge, violin)
- **ML:** scikit-learn (Random Forest, Gradient Boosting, Ridge) + XGBoost
- **Data:** pandas, NumPy, SciPy
- **Persistence:** joblib

---

*Built with ❤️ using Python, Machine Learning & Streamlit*
