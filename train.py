# load packages
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from utils.utils import clean_data, get_preprocessor

# load dataset
data = pd.read_csv("data/raw/loan_status.csv")

# clean dataset
clean_df = clean_data(data, drop_missing=False)

# split dataset
train_df, test_df = train_test_split(
    clean_df,
    test_size=0.2,
    stratify=clean_df["LoanStatus"],
    random_state=42
)

# reset indices
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# map target
train_df["LoanStatus"] = train_df["LoanStatus"].map(lambda x: 1 if x=="Yes" else 0)
test_df["LoanStatus"] = test_df["LoanStatus"].map(lambda x: 1 if x=="Yes" else 0)

# separate features
NUM_FEATURES = [
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'LoanAmountTerm'
    ]

CAT_FEATURES = [
    'Gender',
    'Married',
    'Dependents',
    'Education',
    'SelfEmployed',
    'CreditHistory',
    'PropertyArea'
    ]

LABEL = "LoanStatus"

FEATURES = NUM_FEATURES + CAT_FEATURES

# instantiate preprocessor
preprocessor = get_preprocessor(
    numerical_features=NUM_FEATURES,
    categorical_features=CAT_FEATURES
)

# build model pipeline
estimator = LogisticRegression(
    C=2.3072425041415756,
    class_weight=None,
    max_iter=1000,
    l1_ratio=0,
    solver="saga"
)

classifier = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", estimator)
    ]
)

# separate features and labels
X_train = train_df[FEATURES]
y_train = train_df[LABEL]

X_test = test_df[FEATURES]
y_test = test_df[LABEL]

# train pipeline
classifier.fit(X_train, y_train)

# evaluate model
THRESHOLD = 0.35
y_proba = classifier.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= THRESHOLD).astype("int")

accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
roc_auc = metrics.roc_auc_score(y_test, y_pred)

scores = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc
}

# print metrics
for metric, value in scores.items():
    print(f"{metric}: {value}")

# save score
with open("scores.json", "w") as json_file:
    json.dump(scores, json_file)

# save pipeline
MODEL_DIR = "models"
MODEL_FILEPATH = os.path.join(MODEL_DIR, "classifier.joblib")
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(classifier, MODEL_FILEPATH)


# save test data
DATA_DIR = "data/test"
os.makedirs(DATA_DIR, exist_ok=True)

X_test.to_csv(f"{DATA_DIR}/test_features.csv", index=False)
y_test.to_csv(f"{DATA_DIR}/test_labels.csv", index=False)
