# Step 1: Importing libraries and loading the data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Loading the data and adding labels to the data and checking for NaN values
# Load the data
data = pd.read_csv('expression_file.csv')

# Add labels
data['label'] = data['condition'].map({'tumor': 1, 'normal': 0})

# Check for NaN values
print(f'Number of NaN values in the dataset: {data.isna().sum().sum()}')

# Split data into features and labels
X = data['expression'].values.reshape(-1, 1)
y = data['label']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Step 3: Train the Logistic Regression Model
# Initialize and train the model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate the Model
# Predict on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Step 5: Visualize the Results
# Plot ROC-AUC Curve
y_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Visualize Gene Expression Profiles
plt.figure(figsize=(12, 8))
sns.kdeplot(data[data['label'] == 1]['expression'], label='Tumor', shade=True)
sns.kdeplot(data[data['label'] == 0]['expression'], label='Normal', shade=True)
plt.xlabel('Expression Level')
plt.ylabel('Density')
plt.title('Gene Expression Profiles')
plt.legend()
plt.show()