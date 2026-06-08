# -*- coding: utf-8 -*-
"""modelling_tuned.py

Modelling dengan hyperparameter tuning menggunakan XGBoost dan manual logging ke MLflow.
"""

import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 1. Muat Data yang Sudah Di-preprocessing
df = pd.read_csv("jakarta_house_preprocessing.csv")

# 2. Pisahkan fitur (X) dan target label (y)
X = df.drop('price', axis=1)
y = df['price']

# 3. Split data: training+validation (80%) dan test (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Atur Eksperimen MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Eksperimen_Kaggle_XGBoost")

# 5. Definisikan parameter grid untuk tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Inisialisasi model dasar
xgb_model = xgb.XGBRegressor(random_state=42, eval_metric='rmse')

# 6. Lakukan GridSearchCV dengan cross-validation (5-fold)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                           cv=5, scoring='neg_mean_squared_error',
                           verbose=1, n_jobs=-1)

grid_search.fit(X_train_val, y_train_val)

# 7. Ambil model terbaik dan parameter terbaik
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("Best parameters found:", best_params)
print("Best CV score (neg MSE):", grid_search.best_score_)

# 8. Evaluasi model terbaik pada test set
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test set performance: MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")

# 9. Manual logging ke MLflow
with mlflow.start_run() as run:
    # Log parameter terbaik
    mlflow.log_params(best_params)

    # Log metrik evaluasi test
    mlflow.log_metric("test_mse", mse)
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_r2_score", r2)

    # Log best CV score
    mlflow.log_metric("best_cv_neg_mse", grid_search.best_score_)

    # Simpan model sebagai artifact
    mlflow.xgboost.log_model(best_model, "xgboost_model_tuned")

    # Buat dan simpan feature importance plot sebagai artifact
    importance = best_model.feature_importances_
    features = X.columns
    plt.figure(figsize=(10, 6))
    plt.barh(features, importance)
    plt.xlabel("Feature Importance")
    plt.title("XGBoost Feature Importance (Tuned)")
    plt.tight_layout()
    plt.savefig("feature_importance_tuned.png")
    mlflow.log_artifact("feature_importance_tuned.png")

    print(f"Run ID: {run.info.run_id}")
    print("Semua parameter, metrik, model, dan artifact berhasil dicatat ke MLflow.")