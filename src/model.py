import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics

# Load the data
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, '..', '..', '..', 'Datasets')
dataset_path = os.path.normpath(dataset_path)  # Clean up the path

# Read the CSV files
df = pd.read_csv(os.path.join(dataset_path, 'credit_delinquency_v2.csv'))

# Prepare the data
df = pd.get_dummies(df, prefix_sep="_", drop_first=False, dtype=int)
labels = df["Delinquent"]
df = df.drop(columns="Delinquent")
train_data, test_data, train_labels, test_labels = \
            sklearn.model_selection.train_test_split(df, labels,
            test_size=0.2, shuffle=True, random_state=2025)

# Remove all NaN values
# Fill the NaN values with the mean of the column
train_data = train_data.dropna()
test_data = test_data.dropna()

# Update labels to match the remaining rows
train_labels = train_labels[train_data.index]
test_labels = test_labels[test_data.index]

# Standardize scale for all columns
train_means = train_data.mean()
train_stds = train_data.std()
train_data = (train_data - train_means) / train_stds
test_data  = (test_data  - train_means) / train_stds

print("Final NaN check train:", train_data.isna().sum().sum())
print("Final NaN check test:", test_data.isna().sum().sum())

# Select columns of interest (all columns)
cols = train_data.columns

# Create and train a new logistic regression classifier
model = sklearn.linear_model.LogisticRegression(\
        solver='newton-cg', tol=1e-6)

# Train it with the training data and labels
model.fit(train_data[cols], train_labels)

# Get the prediction probabilities
pred_proba = model.predict_proba(test_data[cols])[:,1]

# Compute a precision & recall graph
precisions, recalls, thresholds = \
    sklearn.metrics.precision_recall_curve(test_labels, pred_proba)
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.legend(loc="center right")
plt.xlabel("Threshold")
plt.axis([0.0, 1.0, 0.0, 1.0])
plt.show()

# Plot a ROC curve (Receiver Operating Characteristic)
fpr, tpr, _ = sklearn.metrics.roc_curve(test_labels, pred_proba)
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.show()

# Compute the area under the ROC curve (ROC AUC)
auc_score = sklearn.metrics.roc_auc_score(test_labels, pred_proba)
print("Test AUC score: {:.4f}".format(auc_score))

# Compute ROC AUC against training data
pred_proba_training = model.predict_proba(train_data[cols])[:,1]

auc_score_training = sklearn.metrics.roc_auc_score(\
    train_labels, pred_proba_training)
print("Train AUC score: {:.4f}".format(auc_score_training))