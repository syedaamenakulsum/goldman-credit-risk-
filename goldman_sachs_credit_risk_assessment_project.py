# Goldman_Sachs_Credit_Risk_Assessment_Project
# This Python script / notebook is a full credit risk assessment project designed for a professional audience.
# It is structured as a Jupyter Notebook (use in .ipynb) with numbered sections: data generation, EDA, preprocessing,
# probability/statistics (PD, LGD, EAD, EL), modeling (Logistic Regression + Random Forest), Bayesian updating,
# evaluation, risk scoring and classification, and a final business summary & README.

# ----- NOTE -----
# To use as a notebook: save this file as `Goldman_Sachs_Credit_Risk_Assessment_Project.ipynb` or paste into a Jupyter cell.
# Each `# %%` marks a new cell when using editors like VS Code or Jupyter Lab.

# %%
# 0. Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# %%
# 1. Synthetic dataset generation
# We create a realistic-looking dataset with borrower-level features and a default flag.
np.random.seed(42)
N = 20000

# Features
age = np.random.randint(21, 70, size=N)
annual_income = np.round(np.random.lognormal(mean=10.5, sigma=0.8, size=N))  # in INR roughly
loan_amount = np.round(np.random.uniform(50000, 2000000, size=N))
loan_term_months = np.random.choice([12, 24, 36, 48, 60], size=N, p=[0.1,0.15,0.4,0.25,0.1])
employment_status = np.random.choice(['salaried','self-employed','unemployed','student'], size=N, p=[0.7,0.18,0.09,0.03])
credit_score = np.clip((np.random.normal(loc=650, scale=70, size=N)).astype(int), 300, 900)
previous_defaults = np.random.poisson(0.1, size=N)
months_since_last_default = np.where(previous_defaults>0, np.random.randint(1,120,size=N), np.nan)

# macro factor - unemployment rate (same for all; could be varied over time)
unemployment_rate = 0.06  # 6%

# We design a latent probability of default using a logistic function combining features
from scipy.special import expit

# base log-odds
log_odds = -5.0
log_odds += (700 - credit_score) * 0.005  # worse score increases odds
log_odds += (loan_amount / 200000) * 0.2
log_odds += (previous_defaults * 1.2)
log_odds += np.where(employment_status=='unemployed', 1.0, 0.0)
log_odds += np.where(employment_status=='student', 0.4, 0.0)
log_odds += (loan_term_months - 12) * 0.01
log_odds += (0.5 * (np.log1p(1e-6 + (1000000/ (annual_income + 1)))))
log_odds += unemployment_rate * 5

pd_prob = expit(log_odds)  # probability of default

# generate defaults
default = np.random.binomial(1, pd_prob)

# LGD: depends on collateral and loan type; we simulate as beta-like variable
lgd = np.clip(np.random.beta(a=2+default, b=5, size=N) , 0.05, 0.95)  # defaulted loans tend to have slightly higher loss

# EAD: outstanding exposure at default - we assume some fraction of loan amount
ead = loan_amount * np.random.uniform(0.6, 1.0, size=N)

# Build DataFrame
df = pd.DataFrame({
    'age': age,
    'annual_income': annual_income,
    'loan_amount': loan_amount,
    'loan_term_months': loan_term_months,
    'employment_status': employment_status,
    'credit_score': credit_score,
    'previous_defaults': previous_defaults,
    'months_since_last_default': months_since_last_default,
    'unemployment_rate': unemployment_rate,
    'pd_prob_true': pd_prob,
    'default': default,
    'lgd': lgd,
    'ead': ead
})

print('Dataset shape:', df.shape)
print('Default rate (synthetic):', df['default'].mean())

# %%
# 2. Exploratory Data Analysis (EDA)
# Basic distributions and relationship with default
print('\n--- EDA Summary ---')
print(df[['annual_income','loan_amount','credit_score','previous_defaults','default']].describe())

# plot histograms (users running notebook will see these)
plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
sns.histplot(df['credit_score'], bins=30, kde=True)
plt.title('Credit Score Distribution')
plt.subplot(2,2,2)
sns.histplot(df['loan_amount']/1e5, bins=30)
plt.title('Loan Amount (lakhs)')
plt.subplot(2,2,3)
sns.histplot(df['annual_income']/1e5, bins=30)
plt.title('Annual Income (lakhs)')
plt.subplot(2,2,4)
sns.countplot(x='previous_defaults', data=df)
plt.title('Previous Defaults Count')
plt.tight_layout()

# Default rates by credit score bucket
df['cs_bucket'] = pd.cut(df['credit_score'], bins=[300,500,600,700,800,900], labels=['300-500','500-600','600-700','700-800','800-900'])
default_by_bucket = df.groupby('cs_bucket')['default'].mean()
print('\nDefault rate by credit score bucket:')
print(default_by_bucket)

# %%
# 3. Preprocessing for modeling
features = ['age','annual_income','loan_amount','loan_term_months','employment_status','credit_score','previous_defaults']
X = df[features]
y = df['default']

# Define preprocessing: scale numeric, one-hot categorical
numeric_features = ['age','annual_income','loan_amount','loan_term_months','credit_score','previous_defaults']
cat_features = ['employment_status']

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
cat_transformer = Pipeline(steps=[('ohe', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', cat_transformer, cat_features)
    ]
)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# %%
# 4. Modeling: Logistic Regression
pipe_log = Pipeline(steps=[('pre', preprocessor), ('clf', LogisticRegression(max_iter=1000))])
pipe_log.fit(X_train, y_train)

# Predict
y_pred_proba_log = pipe_log.predict_proba(X_test)[:,1]
y_pred_log = pipe_log.predict(X_test)

print('\nLogistic Regression Metrics:')
print('ROC-AUC:', roc_auc_score(y_test, y_pred_proba_log))
print('Accuracy:', accuracy_score(y_test, y_pred_log))
print('Precision:', precision_score(y_test, y_pred_log))
print('Recall:', recall_score(y_test, y_pred_log))
print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred_log))

# %%
# 5. Modeling: Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
pipe_rf = Pipeline(steps=[('pre', preprocessor), ('clf', rf)])
pipe_rf.fit(X_train, y_train)

y_pred_proba_rf = pipe_rf.predict_proba(X_test)[:,1]
y_pred_rf = pipe_rf.predict(X_test)

print('\nRandom Forest Metrics:')
print('ROC-AUC:', roc_auc_score(y_test, y_pred_proba_rf))
print('Accuracy:', accuracy_score(y_test, y_pred_rf))
print('Precision:', precision_score(y_test, y_pred_rf))
print('Recall:', recall_score(y_test, y_pred_rf))
print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred_rf))

# %%
# 6. Calibration (important for PD estimates)
calibrator = CalibratedClassifierCV(pipe_rf, method='isotonic', cv=5)
calibrator.fit(X_train, y_train)
cal_y_proba = calibrator.predict_proba(X_test)[:,1]
print('\nCalibrated Random Forest ROC-AUC:', roc_auc_score(y_test, cal_y_proba))

# reliability plot
prob_true, prob_pred = calibration_curve(y_test, cal_y_proba, n_bins=10)
plt.figure(figsize=(6,4))
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1], linestyle='--')
plt.xlabel('Predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration curve (RF isotonic)')

# %%
# 7. PD / LGD / EAD / Expected Loss calculations for test set
# Using calibrated probabilities as PD estimates
test_df = X_test.copy()
test_df = test_df.reset_index(drop=True)
test_df['pd_est'] = cal_y_proba
# Attach true LGD and EAD from original df rows
orig_test_idx = y_test.reset_index(drop=True).index
# To map back, we'll fetch rows by index from original dataset using X_test index
orig_idx = X_test.index

test_df['lgd'] = df.loc[orig_idx,'lgd'].values
test_df['ead'] = df.loc[orig_idx,'ead'].values

# Expected Loss per borrower
test_df['expected_loss'] = test_df['pd_est'] * test_df['lgd'] * test_df['ead']

print('\nSample expected loss (first 5 rows):')
print(test_df[['pd_est','lgd','ead','expected_loss']].head())

# Portfolio-level expected loss
portfolio_EL = test_df['expected_loss'].sum()
print('\nPortfolio Expected Loss (test set):', portfolio_EL)

# %%
# 8. Risk scoring and classification
# Create a risk score between 0 and 1 from pd_est (optionally combine with lgd)
test_df['risk_score'] = test_df['pd_est']  # simple option

# classify into Low/Medium/High using quantiles or thresholds
thresholds = [0.02, 0.08]  # conservative thresholds for this synthetic data

def risk_label(p):
    if p < thresholds[0]:
        return 'Low'
    elif p < thresholds[1]:
        return 'Medium'
    else:
        return 'High'

test_df['risk_label'] = test_df['risk_score'].apply(risk_label)
print('\nRisk label distribution:')
print(test_df['risk_label'].value_counts(normalize=True))

# %%
# 9. Bayesian updating example for PD
# Suppose we have a prior for PD for a borrower segment represented by Beta(alpha,beta)
# We can update this prior with observed defaults to get posterior PD.
from scipy.stats import beta

# Example: prior alpha=1, beta=19 -> prior mean PD = 1/(1+19)=0.05
alpha_prior, beta_prior = 1, 19
observed_defaults = y_test.sum()
observed_nondefaults = (1 - y_test).sum()

alpha_post = alpha_prior + observed_defaults
beta_post = beta_prior + observed_nondefaults
posterior_mean_pd = alpha_post / (alpha_post + beta_post)

print('\nBayesian Updating Example:')
print('Prior PD mean:', alpha_prior/(alpha_prior+beta_prior))
print('Observed defaults (test set):', observed_defaults)
print('Posterior PD mean:', posterior_mean_pd)

# Show posterior credible interval
ci_low, ci_high = beta.ppf([0.025,0.975], alpha_post, beta_post)
print('95% credible interval for PD posterior:', (ci_low, ci_high))

# %%
# 10. Hypothesis testing example: Do borrowers with previous defaults have higher default rate?
from statsmodels.stats.proportion import proportions_ztest

group1 = df[df['previous_defaults']>0]['default']
group2 = df[df['previous_defaults']==0]['default']
count = np.array([group1.sum(), group2.sum()])
nobs = np.array([len(group1), len(group2)])
stat, pval = proportions_ztest(count, nobs)
print('\nHypothesis test: previous defaults vs none')
print('z-stat:', stat, 'p-value:', pval)

# %%
# 11. Model explanations and feature importance (from RF)
# We can extract feature importances (after preprocessing, names change due to OHE)
# To map feature importances back, rebuild transformer on entire dataset
pipe_rf.fit(X, y)  # fit on full data for importance
rf_feat = pipe_rf.named_steps['clf']

# get transformed feature names
ohe = pipe_rf.named_steps['pre'].named_transformers_['cat'].named_steps['ohe']
cat_names = list(ohe.get_feature_names_out(cat_features))
feat_names = numeric_features + cat_names
importances = rf_feat.feature_importances_
feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)
print('\nTop 10 feature importances:')
print(feat_imp.head(10))

# %%
# 12. Business Insights & Next steps (markdown style comments)
# - Use calibrated probabilities for PD to price loans and set capital buffers.
# - Combine PD, LGD, EAD to compute Expected Loss and Unexpected Loss (with VaR) for capital planning.
# - Implement monitoring: track realized defaults vs predicted PD by segment monthly.
# - Consider survival models for time-to-default analysis and macroeconomic scenario testing.

# %%
# 13. Save key outputs and create a report-ready CSV
output = test_df.copy()
output['true_default'] = df.loc[orig_idx,'default'].values
output.to_csv('credit_risk_test_set_results.csv', index=False)
print('\nSaved test set results to credit_risk_test_set_results.csv')

# %%
# 14. README / How to run (as a cell containing instructions)
readme = '''
Goldman Sachs - Credit Risk Assessment Project
=============================================

Files produced:
- credit_risk_test_set_results.csv  : test-set borrower PD, LGD, EAD, Expected Loss, risk label

How to run:
1. Install dependencies: pip install numpy pandas scikit-learn scipy matplotlib seaborn statsmodels
2. Run this notebook in Jupyter or convert to script.
3. Inspect plots for EDA and calibration.

What to include in final report:
- Problem statement and objectives
- Data description and assumptions (synthetic data generation details)
- EDA highlights (default rates by segment)
- Statistical methods (PD estimation, Bayesian updating, hypothesis tests)
- Models tried and evaluation (Logistic vs Random Forest; calibration)
- PD/LGD/EAD calculations and portfolio expected loss
- Business recommendations and limitations
'''
print(readme)

# End of notebook
