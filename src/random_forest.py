import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv(r"C:\Users\trato\OneDrive\Documents\ITESM\SemestreQuant\IA\Module2_FrameworkUse\data\players_data_light-2024_2025.csv")
target_col = "Gls"
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode non-numeric columns using OrdinalEncoder
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
if cat_cols:
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[cat_cols] = oe.fit_transform(X[cat_cols])

#Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_reg = RandomForestRegressor(n_estimators=100, max_features="sqrt", n_jobs=-1,
                              random_state=42, max_depth=8, oob_score=True)
rf_reg.fit(X_train, y_train)
y_pred_train = rf_reg.predict(X_train)
y_pred_test = rf_reg.predict(X_test)
oob = getattr(rf_reg, "oob_score_", None)
if oob is not None:
    print("OOB score:", oob)
print("Train R2:", r2_score(y_train, y_pred_train))
print("Test R2:", r2_score(y_test, y_pred_test))
print("Test MAE:", mean_absolute_error(y_test, y_pred_test))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
top = sorted(zip(X.columns, rf_reg.feature_importances_), key=lambda x: -x[1])[:10]

#Classification test
y_class = (y >= 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=40, stratify=y_class)
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
if cat_cols:
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train[cat_cols] = oe.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = oe.transform(X_test[cat_cols])
rf_clf = RandomForestClassifier(n_estimators=10, max_features="sqrt", n_jobs=-1,
                               random_state=42, max_depth=2, min_samples_leaf=80, oob_score=True)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))