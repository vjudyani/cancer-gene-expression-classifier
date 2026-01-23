"""
generate_sample_data.py
========================
Generates a synthetic gene expression dataset for testing
the cancer classifier pipeline.

Simulates:
  - 200 samples (100 Tumor, 100 Normal)
  - 500 genes
  - Tumor samples have elevated expression in ~50 oncogenes
  - Normal samples have elevated expression in ~50 suppressor genes

Usage:
    python generate_sample_data.py --output data/gene_expression.csv
"""

import argparse
import numpy as np
import pandas as pd

def generate_data(n_samples_per_class=100, n_genes=500, random_state=42):
    """
    Generate synthetic RNA-seq-like gene expression data.

    Parameters
    ----------
    n_samples_per_class : int
    n_genes             : int
    random_state        : int

    Returns
    -------
    df : pd.DataFrame with genes as columns + 'label' column
    """
    np.random.seed(random_state)

    gene_names = [f'GENE_{i:04d}' for i in range(n_genes)]

    # ── Normal samples: baseline expression ──
    normal = np.random.negative_binomial(5, 0.5, (n_samples_per_class, n_genes)).astype(float)

    # ── Tumor samples: elevated expression in first 50 genes (oncogenes) ──
    tumor = np.random.negative_binomial(5, 0.5, (n_samples_per_class, n_genes)).astype(float)
    tumor[:, :50]  += np.random.exponential(8, (n_samples_per_class, 50))   # oncogenes up
    tumor[:, 50:100] -= np.random.uniform(0, 3, (n_samples_per_class, 50))  # suppressors down
    tumor = np.clip(tumor, 0, None)

    # ── Combine ──
    X = np.vstack([normal, tumor])
    y = np.array([0] * n_samples_per_class + [1] * n_samples_per_class)

    sample_names = (
        [f'Normal_{i:03d}' for i in range(n_samples_per_class)] +
        [f'Tumor_{i:03d}'  for i in range(n_samples_per_class)]
    )

    df = pd.DataFrame(X, index=sample_names, columns=gene_names)
    df['label'] = y

    # Shuffle rows
    df = df.sample(frac=1, random_state=random_state)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic gene expression data.')
    parser.add_argument('--output', default='data/gene_expression.csv',
                        help='Output path for generated CSV')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of samples per class (default: 100)')
    parser.add_argument('--genes', type=int, default=500,
                        help='Number of genes (default: 500)')
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"[INFO] Generating synthetic dataset...")
    df = generate_data(n_samples_per_class=args.samples, n_genes=args.genes)
    df.to_csv(args.output)
    print(f"[INFO] Dataset saved to: {args.output}")
    print(f"[INFO] Shape: {df.shape[0]} samples x {df.shape[1]-1} genes")
    print(f"[INFO] Class balance: {df['label'].value_counts().to_dict()}")
