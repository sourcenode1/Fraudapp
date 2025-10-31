import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, f1_score
from fpdf import FPDF  # pip install fpdf

import random

# ======================
# Set Global Random Seed
# ======================
GLOBAL_SEED = 50

# Set seeds
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# ================================
# 1. Load and preprocess data
# ================================
df = pd.read_csv("warranty_claim_fraud_detection_cleaned.csv")
df = df.drop(['PolicyNumber', 'RepNumber'], axis=1)

# Mappings
days_mapping = {'more than 30': 4, '15 to 30': 3, '8 to 15': 2, '1 to 7': 1, 'none': 0, 'NA': 0}
claims_mapping = {'none': 0, '1': 1, '2 to 4': 3, 'more than 4': 5, 'NA': 0}
vehicle_age_mapping = {'new': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
                       '5 years': 5, '6 years': 6, '7 years': 7, 'more than 7': 8, 'NA': -1}
policyholder_age_mapping = {'16 to 17': 1, '18 to 20': 2, '21 to 25': 3, '26 to 30': 4, '31 to 35': 5,
                            '36 to 40': 6, '41 to 50': 7, '51 to 65': 8, 'over 65': 9, 'NA': -1}
suppliments_mapping = {'none': 0, '1 to 2': 2, '3 to 5': 4, 'more than 5': 6, 'NA': 0}
address_change_mapping = {'no change': 0, 'under 6 months': 1, '1 year': 2, '2 to 3 years': 3,
                          '4 to 8 years': 4, 'more than 8 years': 5, 'NA': 0}
number_of_cars_mapping = {'1 vehicle': 1, '2 vehicles': 2, '3 to 4': 3, '5 to 8': 5,
                          'more than 8': 9, 'NA': 0}

df['Days_Policy_Accident'] = df['Days_Policy_Accident'].map(days_mapping).fillna(0)
df['Days_Policy_Claim'] = df['Days_Policy_Claim'].map(days_mapping).fillna(0)
df['PastNumberOfClaims'] = df['PastNumberOfClaims'].map(claims_mapping).fillna(0)
df['AgeOfVehicle'] = df['AgeOfVehicle'].map(vehicle_age_mapping).fillna(-1)
df['AgeOfPolicyHolder'] = df['AgeOfPolicyHolder'].map(policyholder_age_mapping).fillna(-1)
df['NumberOfSuppliments'] = df['NumberOfSuppliments'].map(suppliments_mapping).fillna(0)
df['AddressChange_Claim'] = df['AddressChange_Claim'].map(address_change_mapping).fillna(0)
df['NumberOfCars'] = df['NumberOfCars'].map(number_of_cars_mapping).fillna(0)

# Target / Features
y = df['FraudFound_P']
X = df.drop('FraudFound_P', axis=1)

# Categorical columns
categorical_cols = ['Month', 'DayOfWeek', 'Make', 'AccidentArea',
                    'DayOfWeekClaimed', 'MonthClaimed', 'Sex', 'MaritalStatus',
                    'Fault', 'PolicyType', 'VehicleCategory', 'VehiclePrice',
                    'PoliceReportFiled', 'WitnessPresent', 'AgentType', 'BasePolicy']

for col in categorical_cols:
    X[col] = X[col].astype(str).fillna('NA')

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

train_pool = Pool(X_train, label=y_train, cat_features=categorical_cols)
test_pool = Pool(X_test, label=y_test, cat_features=categorical_cols)

# ================================
# 2. Define Optuna objective
# ================================
def objective(trial):
    params = {
        'iterations': 300,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 20),
        'random_seed': GLOBAL_SEED,
        'early_stopping_rounds': 50,
        'verbose': False,
        'loss_function': 'Logloss',
    }

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    y_prob = model.predict_proba(test_pool)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_f1 = np.max(f1_scores)

    return best_f1

# ================================
# 3. Run Optuna study
# ================================
sampler = optuna.samplers.TPESampler(seed=GLOBAL_SEED)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=30, show_progress_bar=True)

print(f"Best hyperparameters: {study.best_params}")
print(f"Best F1 score from tuning: {study.best_value:.4f}")

# ================================
# 4. Train final model
# ================================
best_params = study.best_params
best_params.update({
    'iterations': 300,
    'random_seed': GLOBAL_SEED,
    'early_stopping_rounds': 50,
    'verbose': False,
    'loss_function': 'Logloss',
    'eval_metric': 'F1',
})
final_model = CatBoostClassifier(**best_params)
final_model.fit(train_pool, eval_set=test_pool, use_best_model=True)

# ================================
# 5. Evaluate final model
# ================================
y_prob = final_model.predict_proba(test_pool)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
best_idx = f1_scores.argmax()
best_thresh = thresholds[best_idx]
y_pred_best = (y_prob > best_thresh).astype(int)

print("\n=== Classification Report (best threshold) ===")
print(classification_report(y_test, y_pred_best))
print(f"Best Threshold: {best_thresh:.4f}")
print(f"Best F1 Score: {f1_scores[best_idx]:.4f}")
print(f"Precision at Best Threshold: {precision[best_idx]:.4f}")
print(f"Recall at Best Threshold: {recall[best_idx]:.4f}")

# ================================
# 6. SHAP Explanation
# ================================
print("Calculating SHAP values (this may take some time)...")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)

# ================================
# 7. Plotting
# ================================
def plot_precision_recall():
    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.scatter(recall[best_idx], precision[best_idx], color='red', label=f'Best threshold: {best_thresh:.2f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pr_curve.png")
    plt.close()

def plot_shap_summary():
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    plt.close()

plot_precision_recall()
plot_shap_summary()

# ================================
# 8. PDF Report
# ================================
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'CatBoost Fraud Detection Model Report', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, text):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 10, text)
        self.ln()

pdf = PDFReport()
pdf.add_page()

# Add tuning summary
pdf.chapter_title("1. Hyperparameter Tuning Results")
pdf.chapter_body(f"Best hyperparameters:\n{study.best_params}\n\nBest F1 score: {study.best_value:.4f}")

# Add classification report
report_str = classification_report(y_test, y_pred_best)
pdf.chapter_title("2. Classification Report (Best Threshold)")
pdf.chapter_body(report_str)
pdf.chapter_body(f"Best Threshold: {best_thresh:.4f}\n"
                 f"Precision: {precision[best_idx]:.4f}\n"
                 f"Recall: {recall[best_idx]:.4f}\n"
                 f"F1 Score: {f1_scores[best_idx]:.4f}")

# Add plots
pdf.chapter_title("3. Plots")
pdf.cell(0, 10, 'Precision-Recall Curve:', 0, 1)
pdf.image("pr_curve.png", w=180)
pdf.ln(10)
pdf.cell(0, 10, 'SHAP Summary Plot:', 0, 1)
pdf.image("shap_summary.png", w=180)

pdf.output("catboost_fraud_detection_report.pdf")
print(" PDF report created: catboost_fraud_detection_report.pdf")

import joblib
joblib.dump(final_model, "final_catboost_model.pkl")
