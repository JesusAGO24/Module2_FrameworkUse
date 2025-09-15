import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Paths
script_loc = Path(__file__).resolve()
repo_root = script_loc.parent.parent
data_path = repo_root / 'data' / 'players_data_light-2024_2025.csv'
report_path = repo_root / 'reports'
report_path.mkdir(exist_ok=True)
report_file = report_path / 'rf_report.pdf'

# Load
print('Reading data from', data_path)
df = pd.read_csv(data_path)

# Target and features
TARGET = 'Gls'
if TARGET not in df.columns:
    raise SystemExit(f"Target {TARGET} not in data")
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Encode categoricals
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
if cat_cols:
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[cat_cols] = oe.fit_transform(X[cat_cols])

# Create train/val/test splits (60/20/20)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25*0.8 = 0.2

# Baseline model
baseline = RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=42)
baseline.fit(X_train, y_train)

# Eval helper
def eval_model(m, Xs, ys):
    preds = m.predict(Xs)
    return {
        'R2': r2_score(ys, preds),
        'MAE': mean_absolute_error(ys, preds),
        'RMSE': np.sqrt(mean_squared_error(ys, preds))
    }

metrics = {
    'train': eval_model(baseline, X_train, y_train),
    'val': eval_model(baseline, X_val, y_val),
    'test': eval_model(baseline, X_test, y_test)
}
print('Baseline metrics:', metrics)

# Regularize via RandomizedSearch on val set
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_leaf': [1, 5, 10, 20],
    'max_features': ['sqrt', 0.3, 0.5]
}
search = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_dist, n_iter=20, cv=3, scoring='r2', n_jobs=-1, random_state=1)
search.fit(X_train, y_train)
print('Best params:', search.best_params_)
best = search.best_estimator_

metrics['train_reg'] = eval_model(best, X_train, y_train)
metrics['val_reg'] = eval_model(best, X_val, y_val)
metrics['test_reg'] = eval_model(best, X_test, y_test)
print('Regularized metrics:', {k: metrics[k] for k in ['train_reg','val_reg','test_reg']})

# Plots
with PdfPages(report_file) as pdf:
    # Metrics table
    fig, ax = plt.subplots(figsize=(6,4))
    ax.axis('off')
    txt = 'Metrics (Baseline vs Regularized)\\n'
    txt += 'Set\tR2_baseline\tR2_reg\tMAE_baseline\tMAE_reg\\n'
    for set_name in ['train','val','test']:
        rb = metrics[set_name]['R2']
        rr = metrics.get(set_name + '_reg', {}).get('R2', '')
        mb = metrics[set_name]['MAE']
        mr = metrics.get(set_name + '_reg', {}).get('MAE', '')
        txt += f"{set_name}\t{rb:.3f}\t{rr:.3f}\t{mb:.3f}\t{mr:.3f}\\n"
    ax.text(0,0.5,txt, fontsize=10, family='monospace')
    pdf.savefig(fig)
    plt.close()

    # Residual plot on test (best)
    preds = best.predict(X_test)
    res = preds - y_test
    fig, ax = plt.subplots()
    ax.scatter(preds, res, alpha=0.5)
    ax.axhline(0, color='red')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Predicted (Regularized)')
    pdf.savefig(fig)
    plt.close()

    # Feature importances
    importances = best.feature_importances_
    idx = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(range(len(idx)), importances[idx][::-1])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([X.columns[i] for i in idx][::-1])
    ax.set_title('Top 15 Feature Importances (Regularized)')
    pdf.savefig(fig)
    plt.close()

print('Report saved to', report_file)
