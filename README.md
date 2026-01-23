# Cancer Gene Expression Classifier 🧬

A machine learning pipeline to classify **tumor vs. normal samples** using RNA-seq gene expression data. This project demonstrates end-to-end bioinformatics analysis — from raw expression data preprocessing through model training, evaluation, and interpretation.

---

## 📌 Project Overview

Gene expression profiling via RNA sequencing (RNA-seq) provides a powerful snapshot of cellular state. This project applies supervised machine learning — specifically **logistic regression** — to distinguish cancerous from normal tissue samples based on their gene expression signatures.

This type of analysis has direct applications in:
- Early cancer detection and diagnosis
- Biomarker discovery
- Personalized medicine and treatment stratification

---

## 🗂️ Repository Structure

```
cancer-gene-expression-classifier/
│
├── data/                        # Input gene expression data (not tracked by git)
├── notebooks/                   # Exploratory data analysis (Jupyter notebooks)
├── src/
│   └── classifier.py            # Main classification script
├── results/                     # Output figures and metrics
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🧪 Dataset

- **Source:** Publicly available gene expression dataset (e.g., TCGA / GEO)
- **Features:** Gene expression values (normalized counts / TPM) across thousands of genes
- **Labels:** Binary — Tumor vs. Normal
- **Format:** CSV / TSV matrix (samples × genes)

> To use your own dataset, place it in the `data/` directory and update the file path in `classifier.py`.

---

## ⚙️ Methods

### 1. Data Preprocessing
- Loaded gene expression matrix using `pandas`
- Handled missing values and removed low-variance genes
- Applied **log2 normalization** to reduce skewness
- Scaled features using `StandardScaler` from scikit-learn

### 2. Exploratory Data Analysis
- PCA visualization to assess sample clustering by class
- Heatmap of top differentially expressed genes

### 3. Model Training
- Algorithm: **Logistic Regression** (scikit-learn)
- Train/test split: 80/20 stratified split
- Hyperparameter tuning via cross-validation

### 4. Model Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Confusion matrix visualization
- Feature importance: top genes contributing to classification

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | ~XX% |
| Precision | ~XX% |
| Recall | ~XX% |
| F1-Score | ~XX% |
| ROC-AUC | ~XX |

> *(Update this table with your actual results)*

---

## 🛠️ Requirements

```bash
Python >= 3.8
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Clone the repository
```bash
git clone https://github.com/vjudyani/cancer-gene-expression-classifier.git
cd cancer-gene-expression-classifier
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the classifier
```bash
python src/classifier.py --input data/gene_expression.csv --output results/
```

### Or explore the notebook
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

---

## 🔬 Biological Context

Tumor samples exhibit distinct transcriptomic signatures compared to normal tissue — including upregulation of oncogenes and downregulation of tumor suppressor genes. Logistic regression, while simple, is highly interpretable and clinically relevant because the model coefficients directly indicate which genes are the strongest predictors of malignancy.

This makes it particularly useful in a translational research setting where biological explainability is as important as predictive accuracy.

---

## 🔭 Future Improvements

- [ ] Expand to multi-class classification (multiple cancer types)
- [ ] Implement Random Forest and XGBoost for performance comparison
- [ ] Add feature selection using LASSO regularization
- [ ] Integrate with TCGA data via the GDC API
- [ ] Build a Snakemake pipeline for reproducibility

---

## 👩‍💻 Author

**Vedika Judyani**
MS Bioinformatics | Bioinformatics Analyst
[LinkedIn](https://www.linkedin.com/in/vedika-judyani-a19011128/) | [GitHub](https://github.com/vjudyani)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 📚 References

- [TCGA — The Cancer Genome Atlas](https://www.cancer.gov/tcga)
- [GEO — Gene Expression Omnibus](https://www.ncbi.nlm.nih.gov/geo/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
