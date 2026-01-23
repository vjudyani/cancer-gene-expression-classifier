"""
Cancer Gene Expression Classifier
===================================
Classifies tumor vs. normal samples using logistic regression
on RNA-seq gene expression data.

Author: Vedika Judyani
MS Bioinformatics
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_data(filepath):
    """
    Load gene expression matrix from a CSV file.

    Expected format:
        - Rows    = samples
        - Columns = genes (features) + one 'label' column
        - Label   : 1 = Tumor, 0 = Normal

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    X : pd.DataFrame  — gene expression features
    y : pd.Series     — binary labels (1=Tumor, 0=Normal)
    """
    print(f"[INFO] Loading data from: {filepath}")
    df = pd.read_csv(filepath, index_col=0)

    if 'label' not in df.columns:
        raise ValueError("Dataset must contain a 'label' column (1=Tumor, 0=Normal).")

    X = df.drop(columns=['label'])
    y = df['label']

    print(f"[INFO] Dataset shape : {X.shape[0]} samples x {X.shape[1]} genes")
    print(f"[INFO] Class balance : {y.value_counts().to_dict()}")
    return X, y


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(X):
    """
    Clean and normalize gene expression data.

    Steps:
        1. Drop genes with >20% missing values
        2. Fill remaining NaNs with column median
        3. Remove low-variance genes (variance < 0.01)
        4. Apply log2(x + 1) normalization

    Parameters
    ----------
    X : pd.DataFrame

    Returns
    -------
    X_processed : pd.DataFrame
    """
    print("\n[INFO] Preprocessing data...")

    # Drop high-missingness genes
    missing_thresh = 0.2
    X = X.loc[:, X.isnull().mean() < missing_thresh]

    # Fill remaining NaNs with median
    X = X.fillna(X.median())

    # Remove low-variance genes
    variances = X.var()
    X = X.loc[:, variances >= 0.01]
    print(f"[INFO] Genes remaining after variance filter: {X.shape[1]}")

    # Log2 normalization (common for RNA-seq data)
    X = np.log2(X + 1)

    return X


# ─────────────────────────────────────────────
# 3. FEATURE SCALING
# ─────────────────────────────────────────────

def scale_features(X_train, X_test):
    """
    Standardize features using training set statistics only.
    Prevents data leakage from test set into scaler.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame

    Returns
    -------
    X_train_scaled, X_test_scaled : np.ndarray
    scaler : fitted StandardScaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────

def train_model(X_train, y_train):
    """
    Train a logistic regression classifier with cross-validation.

    Uses L2 regularization and liblinear solver, suitable for
    high-dimensional genomic data.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : pd.Series

    Returns
    -------
    model : fitted LogisticRegression
    cv_scores : np.ndarray of cross-validation AUC scores
    """
    print("\n[INFO] Training Logistic Regression model...")

    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='liblinear',
        max_iter=1000,
        random_state=42
    )

    # 5-fold stratified cross-validation on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"[INFO] Cross-validation AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    model.fit(X_train, y_train)
    return model, cv_scores


# ─────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, output_dir):
    """
    Evaluate the trained model and save all metrics and plots.

    Parameters
    ----------
    model      : fitted LogisticRegression
    X_test     : np.ndarray
    y_test     : pd.Series
    output_dir : str — folder to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # ── Print metrics ──
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_pred_prob)

    print("\n" + "="*45)
    print("           MODEL EVALUATION RESULTS")
    print("="*45)
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1-Score   : {f1:.4f}")
    print(f"  ROC-AUC    : {auc:.4f}")
    print("="*45)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Tumor']))

    # ── Save metrics to CSV ──
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'Score':  [acc, prec, rec, f1, auc]
    })
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)

    # ── Plot 1: Confusion Matrix ──
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Tumor'],
                yticklabels=['Normal', 'Tumor'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved.")

    # ── Plot 2: ROC Curve ──
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='steelblue', lw=2,
             label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — Tumor vs. Normal', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150)
    plt.close()
    print(f"[INFO] ROC curve saved.")

    return acc, prec, rec, f1, auc


# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def plot_top_genes(model, feature_names, output_dir, top_n=20):
    """
    Plot the top N most important genes based on logistic
    regression coefficients.

    Positive coefficients → associated with Tumor
    Negative coefficients → associated with Normal

    Parameters
    ----------
    model         : fitted LogisticRegression
    feature_names : list of gene names
    output_dir    : str
    top_n         : int, number of top genes to display
    """
    coefficients = model.coef_[0]
    gene_importance = pd.Series(coefficients, index=feature_names)

    # Get top positive (tumor-associated) and negative (normal-associated)
    top_positive = gene_importance.nlargest(top_n // 2)
    top_negative = gene_importance.nsmallest(top_n // 2)
    top_genes = pd.concat([top_positive, top_negative]).sort_values()

    colors = ['#d73027' if v > 0 else '#4575b4' for v in top_genes.values]

    plt.figure(figsize=(10, 7))
    bars = plt.barh(top_genes.index, top_genes.values, color=colors)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.xlabel('Logistic Regression Coefficient', fontsize=12)
    plt.title(f'Top {top_n} Most Influential Genes\n'
              f'(Red = Tumor-associated | Blue = Normal-associated)',
              fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_genes.png'), dpi=150)
    plt.close()
    print(f"[INFO] Top genes plot saved.")

    # Save to CSV
    top_genes.to_frame(name='coefficient').to_csv(
        os.path.join(output_dir, 'top_genes.csv')
    )


# ─────────────────────────────────────────────
# 7. PCA VISUALIZATION
# ─────────────────────────────────────────────

def plot_pca(X_scaled, y, output_dir):
    """
    Reduce to 2 principal components and plot sample clustering
    colored by class label (Tumor vs Normal).

    Parameters
    ----------
    X_scaled   : np.ndarray — scaled gene expression matrix
    y          : pd.Series  — labels
    output_dir : str
    """
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        'PC1': components[:, 0],
        'PC2': components[:, 1],
        'Label': y.values
    })

    colors = {0: '#4575b4', 1: '#d73027'}
    label_names = {0: 'Normal', 1: 'Tumor'}

    plt.figure(figsize=(8, 6))
    for label, group in pca_df.groupby('Label'):
        plt.scatter(group['PC1'], group['PC2'],
                    c=colors[label], label=label_names[label],
                    alpha=0.7, edgecolors='white', s=60)

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    plt.title('PCA — Gene Expression Sample Clustering', fontsize=14, fontweight='bold')
    plt.legend(title='Sample Type', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_plot.png'), dpi=150)
    plt.close()
    print(f"[INFO] PCA plot saved.")


# ─────────────────────────────────────────────
# 8. MAIN PIPELINE
# ─────────────────────────────────────────────

def main(input_path, output_dir):
    """
    Full pipeline: load → preprocess → train → evaluate → visualize.
    """
    print("\n" + "="*45)
    print("  CANCER GENE EXPRESSION CLASSIFIER")
    print("="*45)

    # Load
    X, y = load_data(input_path)

    # Preprocess
    X = preprocess(X)

    # Train/test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # PCA visualization (on all scaled data for overview)
    X_all_scaled = scaler.transform(X)
    plot_pca(X_all_scaled, y, output_dir)

    # Train
    model, cv_scores = train_model(X_train_scaled, y_train)

    # Evaluate
    evaluate_model(model, X_test_scaled, y_test, output_dir)

    # Feature importance
    plot_top_genes(model, X.columns.tolist(), output_dir, top_n=20)

    print(f"\n[DONE] All results saved to: {output_dir}/")


# ─────────────────────────────────────────────
# COMMAND LINE ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Classify tumor vs. normal samples from gene expression data.'
    )
    parser.add_argument('--input',  required=True,
                        help='Path to gene expression CSV file')
    parser.add_argument('--output', default='results/',
                        help='Directory to save output plots and metrics')
    args = parser.parse_args()

    main(args.input, args.output)
