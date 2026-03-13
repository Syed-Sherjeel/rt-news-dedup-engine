import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the CSV data
bgi_df = pd.read_csv('../../../data/bgi_large.csv', sep='|')
gold_df = pd.read_csv('gold.csv')

print("BGI Dataset shape:", bgi_df.shape)
print("Gold Dataset shape:", gold_df.shape)

# Merge on 'id' - bgi_df provides predictions (verdict), gold_df provides ground truth (llm_verdict)
merged_df = bgi_df[['id', 'verdict']].merge(
    gold_df[['id', 'llm_verdict']], 
    on='id', 
    how='inner'
)

print("\nMerged Dataset shape:", merged_df.shape)
print("\nFirst few rows:")
print(merged_df.head(10))
print("\nPrediction (verdict) value counts:")
print(merged_df['verdict'].value_counts())
print("\nGround Truth (llm_verdict) value counts:")
print(merged_df['llm_verdict'].value_counts())

# ============================================================================
# SCENARIO 1: PARAPHRASE ARE POSITIVE (counted as duplicates)
# ============================================================================
print("\n" + "="*80)
print("SCENARIO 1: PARAPHRASE counts as DUPLICATE (Positive)")
print("="*80)

# Convert verdicts: NEW -> 0 (Negative), DUPE or PARAPHRASE -> 1 (Positive)
y_true_scenario1 = (merged_df['llm_verdict'] != 'NEW').astype(int)  # Ground truth
y_pred_scenario1 = (merged_df['verdict'] != 'NEW').astype(int)  # Prediction

cm1 = confusion_matrix(y_true_scenario1, y_pred_scenario1)

print("\nConfusion Matrix (PARAPHRASE = DUPLICATE):")
print(cm1)
print("\nClassification Report:")
print(classification_report(y_true_scenario1, y_pred_scenario1, target_names=['NEW', 'DUPE/PARAPHRASE']))

# ============================================================================
# SCENARIO 2: PARAPHRASE ARE NEGATIVE (only DUPE counts as positive)
# ============================================================================
print("\n" + "="*80)
print("SCENARIO 2: PARAPHRASE counts as NEW (Negative)")
print("="*80)

# Convert verdicts: DUPE -> 1 (Positive), NEW or PARAPHRASE -> 0 (Negative)
y_true_scenario2 = (merged_df['llm_verdict'] == 'DUPE').astype(int)  # Ground truth
y_pred_scenario2 = (merged_df['verdict'] == 'DUPE').astype(int)  # Prediction

cm2 = confusion_matrix(y_true_scenario2, y_pred_scenario2)

print("\nConfusion Matrix (PARAPHRASE = NEW):")
print(cm2)
print("\nClassification Report:")
print(classification_report(y_true_scenario2, y_pred_scenario2, target_names=['NEW/PARAPHRASE', 'DUPE']))

# ============================================================================
# VISUALIZATION
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scenario 1
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            xticklabels=['NEW', 'DUPE/PARAPHRASE'], 
            yticklabels=['NEW', 'DUPE/PARAPHRASE'],
            cbar_kws={'label': 'Count'})
axes[0].set_title('Scenario 1: PARAPHRASE = DUPLICATE\n(Positive Class)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Ground Truth (llm_verdict)', fontweight='bold')
axes[0].set_xlabel('Prediction (verdict)', fontweight='bold')

# Scenario 2
sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=['NEW/PARAPHRASE', 'DUPE'],
            yticklabels=['NEW/PARAPHRASE', 'DUPE'],
            cbar_kws={'label': 'Count'})
axes[1].set_title('Scenario 2: PARAPHRASE = NEW\n(Negative Class)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Ground Truth (llm_verdict)', fontweight='bold')
axes[1].set_xlabel('Prediction (verdict)', fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrices_bgi.png', dpi=300, bbox_inches='tight')
print("\n✓ Confusion matrices saved as 'confusion_matrices_bgi.png'")
plt.show()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\nScenario 1 - PARAPHRASE = DUPLICATE:")
print(f"  True Negatives (NEW, predicted NEW): {cm1[0,0]}")
print(f"  False Positives (NEW, predicted DUPE/PARA): {cm1[0,1]}")
print(f"  False Negatives (DUPE/PARA, predicted NEW): {cm1[1,0]}")
print(f"  True Positives (DUPE/PARA, predicted DUPE/PARA): {cm1[1,1]}")

print("\nScenario 2 - PARAPHRASE = NEW:")
print(f"  True Negatives (NEW/PARA, predicted NEW/PARA): {cm2[0,0]}")
print(f"  False Positives (NEW/PARA, predicted DUPE): {cm2[0,1]}")
print(f"  False Negatives (DUPE, predicted NEW/PARA): {cm2[1,0]}")
print(f"  True Positives (DUPE, predicted DUPE): {cm2[1,1]}")