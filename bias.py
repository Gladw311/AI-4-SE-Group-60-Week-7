import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# 1. Load the COMPAS dataset using AIF360's built-in dataset
# `protected_attribute_names` and `privileged_classes` are set to focus on race, with 'Caucasian' as privileged
# `label_names` is 'two_year_recid' (0: no recidivism, 1: recidivism)
# `favorable_label` is 0 (no recidivism is favorable)
compas_dataset = CompasDataset(
    protected_attribute_names=['race'],
    privileged_classes=[['Caucasian']],
    features_to_drop=['sex', 'c_charge_degree', 'c_charge_desc', 'age_cat', 'decile_score', 'score_text']
    # Dropping some features to simplify, 'sex' is also a protected attribute but focusing on 'race' as per prompt.
    # 'decile_score' is the model output, 'score_text' derived from it.
)

# Split the dataset into training and testing
dataset_orig_train, dataset_orig_test = compas_dataset.split([0.7], shuffle=True)

# Define privileged and unprivileged groups
privileged_groups = [{'race': 1}] # 'Caucasian' mapped to 1 after one-hot encoding by AIF360
unprivileged_groups = [{'race': 0}] # Other races mapped to 0

# 2. Train an unmitigated Logistic Regression model
# Prepare data for sklearn model
X_train = dataset_orig_train.features
y_train = dataset_orig_train.labels.ravel()
X_test = dataset_orig_test.features
y_test = dataset_orig_test.labels.ravel()

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1] # Probability of recidivism (unfavorable label)

# Add predictions to the AIF360 test dataset object
dataset_orig_test_pred = dataset_orig_test.copy()
dataset_orig_test_pred.labels = y_pred
dataset_orig_test_pred.scores = y_proba # For metrics that use scores/probabilities

# 3. Bias Detection: Calculate False Positive Rates and Difference
metric_orig_test = ClassificationMetric(
    dataset_orig_test,
    dataset_orig_test_pred,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

# Calculate False Positive Rates (FPR) for privileged and unprivileged groups
fpr_privileged = metric_orig_test.false_positive_rate(privileged=True)
fpr_unprivileged = metric_orig_test.false_positive_rate(privileged=False)
fpr_difference = metric_orig_test.false_positive_rate_difference(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)

print("--- Unmitigated Model Bias Metrics ---")
print(f"False Positive Rate (Privileged Group - Caucasian): {fpr_privileged:.4f}")
print(f"False Positive Rate (Unprivileged Group - Other): {fpr_unprivileged:.4f}")
print(f"False Positive Rate Difference (Unprivileged - Privileged): {fpr_difference:.4f}")
print("\n")

# 4. Visualization of False Positive Rates
group_names = ['Caucasian', 'Other Races']
fpr_values = [fpr_privileged, fpr_unprivileged]

plt.figure(figsize=(8, 6))
sns.barplot(x=group_names, y=fpr_values, palette='viridis')
plt.title('False Positive Rate Disparity in Unmitigated COMPAS Model')
plt.ylabel('False Positive Rate')
plt.xlabel('Racial Group')
plt.ylim(0, max(fpr_values) * 1.2) # Adjust y-axis limit for better visualization
for index, value in enumerate(fpr_values):
    plt.text(index, value + 0.01, f'{value:.4f}', ha='center')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# 5. Bias Mitigation (Reweighing - Pre-processing technique)
# Reweighing will re-weight the training examples to ensure fairness
RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
dataset_transf_train = RW.fit_transform(dataset_orig_train)

# Train a new model on the re-weighted dataset
model_reweighted = LogisticRegression(solver='liblinear', random_state=42)
# Need to use the sample_weights from the reweighted dataset
model_reweighted.fit(X_train, y_train, sample_weight=dataset_transf_train.instance_weights)

# Make predictions with the re-weighted model
y_pred_reweighted = model_reweighted.predict(X_test)
y_proba_reweighted = model_reweighted.predict_proba(X_test)[:, 1]

# Add predictions to a new AIF360 test dataset object
dataset_transf_test_pred = dataset_orig_test.copy()
dataset_transf_test_pred.labels = y_pred_reweighted
dataset_transf_test_pred.scores = y_proba_reweighted

# 6. Evaluate mitigated model bias metrics
metric_transf_test = ClassificationMetric(
    dataset_orig_test, # Original ground truth
    dataset_transf_test_pred, # Predictions from mitigated model
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

fpr_privileged_mitigated = metric_transf_test.false_positive_rate(privileged=True)
fpr_unprivileged_mitigated = metric_transf_test.false_positive_rate(privileged=False)
fpr_difference_mitigated = metric_transf_test.false_positive_rate_difference(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)

print("--- Mitigated Model (Reweighing) Bias Metrics ---")
print(f"False Positive Rate (Privileged Group - Caucasian): {fpr_privileged_mitigated:.4f}")
print(f"False Positive Rate (Unprivileged Group - Other): {fpr_unprivileged_mitigated:.4f}")
print(f"False Positive Rate Difference (Unprivileged - Privileged): {fpr_difference_mitigated:.4f}")
print("\n")

# 7. Visualization of False Positive Rates (Mitigated vs. Unmitigated)
# Create a DataFrame for plotting
data = {
    'Model': ['Unmitigated'] * 2 + ['Mitigated'] * 2,
    'Group': ['Caucasian', 'Other Races'] * 2,
    'FPR': [fpr_privileged, fpr_unprivileged, fpr_privileged_mitigated, fpr_unprivileged_mitigated]
}
df_fpr = pd.DataFrame(data)

plt.figure(figsize=(10, 7))
sns.barplot(x='Group', y='FPR', hue='Model', data=df_fpr, palette='coolwarm')
plt.title('False Positive Rate Disparity: Unmitigated vs. Reweighed Model')
plt.ylabel('False Positive Rate')
plt.xlabel('Racial Group')
plt.ylim(0, max(df_fpr['FPR']) * 1.2)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()