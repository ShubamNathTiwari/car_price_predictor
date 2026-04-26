"""
model_trainer.py - ML Model Training Script for Car Price Predictor
Trains multiple models and saves the best one
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def load_and_preprocess(filepath='car_data.csv'):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # Drop duplicates and nulls
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # Feature: Car Age
    df['Car_Age'] = 2024 - df['Year']

    # Convert numeric columns — handle cases where values may contain units
    for col in ['Mileage', 'Engine', 'Max_Power']:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.extract(r'([\d.]+)', expand=False).astype(float)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Seats'] = pd.to_numeric(df['Seats'], errors='coerce')

    # Drop rows where numeric conversion failed
    df.dropna(subset=['Mileage', 'Engine', 'Max_Power', 'Seats'], inplace=True)

    # Encode categorical columns (including Owner which is a string like "First Owner")
    le_dict = {}
    cat_cols = ['Car_Name', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']
    for col in cat_cols:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    return df, le_dict


def get_features_target(df):
    # FIX: Include Car_Name_enc so the car brand affects the prediction.
    # Selling_Price is the TARGET — never include it in features (data leakage).
    feature_cols = [
        'Car_Name_enc',
        'Car_Age', 'Kms_Driven', 'Mileage', 'Engine', 'Max_Power', 'Seats',
        'Fuel_Type_enc', 'Seller_Type_enc', 'Transmission_enc', 'Owner_enc'
    ]
    X = df[feature_cols]
    y = df['Selling_Price']
    return X, y, feature_cols


def train_models(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
        'Ridge Regression': Ridge(alpha=1.0),
    }

    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, verbosity=0)

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results[name] = {
            'R2 Score': round(r2 * 100, 2),
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'model': model
        }
        trained_models[name] = model
        print(f"✅ {name}: R² = {r2*100:.2f}%, MAE = {mae:.4f}, RMSE = {rmse:.4f}")

    # Pick best by R2
    best_name = max(results, key=lambda k: results[k]['R2 Score'])
    print(f"\n🏆 Best Model: {best_name} (R² = {results[best_name]['R2 Score']}%)")

    return results, trained_models, best_name


def save_artifacts(trained_models, le_dict, feature_cols, results, df):
    os.makedirs('model_artifacts', exist_ok=True)

    for name, model in trained_models.items():
        safe_name = name.replace(' ', '_').lower()
        joblib.dump(model, f'model_artifacts/{safe_name}.pkl')

    joblib.dump(le_dict, 'model_artifacts/label_encoders.pkl')
    joblib.dump(feature_cols, 'model_artifacts/feature_cols.pkl')

    # Save model results (strip model objects)
    results_clean = {k: {m: v for m, v in val.items() if m != 'model'} for k, val in results.items()}
    joblib.dump(results_clean, 'model_artifacts/model_results.pkl')

    # Save column stats for UI
    stats = {
        'car_names': sorted(df['Car_Name'].unique().tolist()),
        'fuel_types': df['Fuel_Type'].unique().tolist(),
        'seller_types': df['Seller_Type'].unique().tolist(),
        'transmissions': df['Transmission'].unique().tolist(),
        'owner_values': sorted(df['Owner'].unique().tolist()),   # FIX: store actual string values
        'year_min': int(df['Year'].min()),
        'year_max': int(df['Year'].max()),
        'kms_min': int(df['Kms_Driven'].min()),
        'kms_max': int(df['Kms_Driven'].max()),
        'mileage_min': float(df['Mileage'].min()),
        'mileage_max': float(df['Mileage'].max()),
        'engine_min': int(df['Engine'].min()),
        'engine_max': int(df['Engine'].max()),
        'max_power_min': float(df['Max_Power'].min()),
        'max_power_max': float(df['Max_Power'].max()),
        'seats_values': sorted(df['Seats'].dropna().unique().astype(int).tolist()),
        'price_min': float(df['Selling_Price'].min()),
        'price_max': float(df['Selling_Price'].max()),
        'price_mean': float(df['Selling_Price'].mean()),
        'total_records': len(df),
    }
    joblib.dump(stats, 'model_artifacts/dataset_stats.pkl')

    print("\n📦 All artifacts saved to model_artifacts/")


def main():
    print("=" * 60)
    print("  🚗 CAR PRICE PREDICTOR - MODEL TRAINING")
    print("=" * 60)

    print("\n📊 Loading & preprocessing data...")
    df, le_dict = load_and_preprocess('car_data.csv')
    print(f"   Dataset shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")

    X, y, feature_cols = get_features_target(df)
    print(f"   Features used: {feature_cols}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    print("\n🤖 Training models...\n")
    results, trained_models, best_name = train_models(X_train, X_test, y_train, y_test)

    save_artifacts(trained_models, le_dict, feature_cols, results, df)

    print("\n✅ Training complete! Run: streamlit run app.py")
    return True


if __name__ == '__main__':
    main()
