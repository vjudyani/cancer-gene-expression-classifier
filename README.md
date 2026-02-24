# Cancer Gene Expression Classifier

A machine learning pipeline to classify **tumor vs. normal samples** using RNA-seq gene expression data. This project demonstrates end-to-end bioinformatics analysis — from raw expression data preprocessing through model training, evaluation, and interpretation.

---

##  Project Overview

Gene expression profiling via RNA sequencing (RNA-seq) provides a powerful snapshot of cellular state. This project applies supervised machine learning — specifically **logistic regression** — to distinguish cancerous from normal tissue samples based on their gene expression signatures.

This type of analysis has direct applications in:
- Early cancer detection and diagnosis
- Biomarker discovery
- Personalized medicine and treatment stratification

---

## Repository Structure

```
cancer-gene-expression-classifier/
│
├── data/                            # Input gene expression data (not tracked by git)
│   └── gene_expression.csv          # Sample matrix (samples × genes + label column)
├── src/
│   ├── classifier.py                # Main classification pipeline
│   └── generate_sample_data.py      # Script to generate synthetic test data
├── results/                         # Output figures and metrics (auto-generated)
│   ├── pca_plot.png                 # PCA sample clustering
│   ├── confusion_matrix.png         # Confusion matrix heatmap
│   ├── roc_curve.png                # ROC curve
│   ├── top_genes.png                # Top 20 most influential genes
│   └── metrics.csv                  # All evaluation metrics
├── requirements.txt                 # Python dependencies
└── README.md
```

---

##  Dataset

- **Source:** Publicly available gene expression dataset (e.g., TCGA / GEO)
- **Features:** Gene expression values (normalized counts / TPM) across hundreds of genes
- **Labels:** Binary — Tumor (1) vs. Normal (0)
- **Format:** CSV matrix (samples × genes) with a `label` column

> To generate a synthetic dataset for testing, run:
> ```bash
> python src/generate_sample_data.py --output data/gene_expression.csv
> ```

---

##  Methods

### 1. Data Preprocessing
- Loaded gene expression matrix using `pandas`
- Dropped genes with >20% missing values
- Filled remaining NaNs with column median
- Removed low-variance genes (variance < 0.01)
- Applied **log2(x + 1) normalization** to reduce expression skewness

### 2. Exploratory Data Analysis
- PCA (2 components) to visualize sample clustering by class label
- Identifies whether tumor and normal samples are separable in expression space

### 3. Model Training
- Algorithm: **Logistic Regression** with L2 regularization
- Train/test split: **80/20 stratified** split to preserve class balance
- **5-fold stratified cross-validation** on training set for robust AUC estimation
- Solver: `liblinear` (suitable for high-dimensional genomic data)

### 4. Model Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Confusion matrix and ROC curve visualizations saved to `results/`
- Feature importance: top 20 genes by logistic regression coefficient magnitude

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 1.00 |
| Precision | 1.00 |
| Recall | 1.00 |
| F1-Score | 1.00 |
| ROC-AUC | 1.00 |
| CV AUC (5-fold) | 1.000 ± 0.000 |

> Results above are from a **synthetic dataset** designed to demonstrate the pipeline.
> On real-world RNA-seq data (e.g., TCGA), expect AUC in the range of 0.90–0.98 depending
> on cancer type and dataset quality.

---

##  Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.12.0
jupyter>=1.0.0
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/vjudyani/cancer-gene-expression-classifier.git
cd cancer-gene-expression-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate sample data (or use your own)
```bash
python src/generate_sample_data.py --output data/gene_expression.csv
```

### 4. Run the classifier
```bash
python src/classifier.py --input data/gene_expression.csv --output results/
```

Output files will be saved automatically to `results/`.

---

##  Biological Context

Tumor samples exhibit distinct transcriptomic signatures compared to normal tissue — including upregulation of oncogenes and downregulation of tumor suppressor genes. Logistic regression, while simple, is highly interpretable and clinically relevant because the model coefficients directly indicate which genes are the strongest predictors of malignancy.

This makes it particularly useful in a translational research setting where biological explainability is as important as predictive accuracy.

---



## Author

**Vedika Judyani**
MS Bioinformatics | Bioinformatics Analyst
[LinkedIn](https://www.linkedin.com/in/vedika-judyani-a19011128/) | [GitHub](https://github.com/vjudyani)

---

---

## References

- [TCGA — The Cancer Genome Atlas](https://www.cancer.gov/tcga)
- [GEO — Gene Expression Omnibus](https://www.ncbi.nlm.nih.gov/geo/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
