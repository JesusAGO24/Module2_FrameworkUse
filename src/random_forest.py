import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv(r"C:\Users\trato\OneDrive\Documents\ITESM\SemestreQuant\IA\Module2_FrameworkUse\data\players_data_light-2024_2025.csv")
target_col = "Gls"
X = df.drop(columns=[target_col])
y = df[target_col]

# numeric features only 
numeric_X = X.select_dtypes(include=[np.number])
if numeric_X.shape[1] != X.shape[1]:
	dropped = list(set(X.columns) - set(numeric_X.columns))
	print(f"Warning: dropping non-numeric columns: {dropped}")
X = numeric_X

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# tunear
rf = RandomForestRegressor(n_estimators=100, max_features="sqrt", n_jobs=-1,
						   random_state=42, max_depth=8, oob_score=True)

rf.fit(X_train, y_train)

# Predictions and metrics
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

oob = getattr(rf, "oob_score_", None)
if oob is not None:
	print("OOB score:", oob)

print("Train R2:", r2_score(y_train, y_pred_train))
print("Test R2:", r2_score(y_test, y_pred_test))
print("Test MAE:", mean_absolute_error(y_test, y_pred_test))
print("Test RMSE:", mean_squared_error(y_test, y_pred_test, squared=False))

# Feature importances (impurity-based)
top = sorted(zip(X.columns, rf.feature_importances_), key=lambda x: -x[1])[:10]
print("Top features:", top)

print("First 5 test preds:", y_pred_test[:5])
print("First 5 test actual:", y_test.values[:5])