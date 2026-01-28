# ml_code.py

import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
import joblib
from scipy.sparse import hstack, csr_matrix
from itertools import cycle

MODEL_DIR = "models"
Path(MODEL_DIR).mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {df.shape}")
    return df

def preprocess_data(df, is_train=True):
    df = df.copy()

    if is_train:
        df = df.dropna()
    else:
        df.fillna(df.mean(numeric_only=True), inplace=True)

    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)

    cat_cols = [
        'Listing Confirmation Method', 'Direction', 'Parking Availability',
        'Real Estate Agency', 'Brokerage Platform', 'Listing Weekday'
    ]

    if is_train and 'False Listing' in df.columns:
        cat_cols.append('False Listing')

    if is_train:
        le_dict = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
        joblib.dump(le_dict, os.path.join(MODEL_DIR, "label_encoders1.pkl"))
    else:
        le_dict = joblib.load(os.path.join(MODEL_DIR, "label_encoders1.pkl"))
        for col in cat_cols:
            le = le_dict[col]
            known_classes = set(le.classes_)
            df[col] = df[col].astype(str).apply(lambda x: x if x in known_classes else 'Unknown')
            if 'Unknown' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'Unknown')
            df[col] = le.transform(df[col])

    if is_train:
        le_target = LabelEncoder()
        df['False Listing'] = le_target.fit_transform(df['False Listing'].astype(str))
        joblib.dump(le_target, os.path.join(MODEL_DIR, 'target_label_encoder1.pkl'))
        y = df['False Listing']
    else:
        y = None

    num_cols = [
        'Deposit Amount', 'Monthly Rent', 'Exclusive Area', 'Unit Floor',
        'Total Floors', 'Number of Rooms', 'Number of Bathrooms',
        'Total Parking Spaces', 'Maintenance Fee', 'Listing Year',
        'Listing Month', 'Listing Day', 'Days Since Listed'
    ]

    if is_train:
        scaler = StandardScaler()
        num_data = scaler.fit_transform(df[num_cols])
        joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler1.pkl'))
    else:
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler1.pkl'))
        num_data = scaler.transform(df[num_cols])

    cat_data = df[cat_cols].values
    X = hstack([csr_matrix(cat_data), csr_matrix(num_data)])

    return df, X, y

def perform_eda(df):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    sns.distplot(df['Monthly Rent'], kde=True)
    plt.title("Monthly Rent Distribution")

    plt.subplot(2, 3, 2)
    sns.scatterplot(x='Total Floors', y='Number of Rooms', data=df)
    plt.title("Rooms vs Floors")

    plt.subplot(2, 3, 3)
    sns.barplot(x='False Listing', y='Monthly Rent', data=df)
    plt.title("Rent vs Label")

    plt.subplot(2, 3, 4)
    top_cats = df['Parking Availability'].value_counts().index[:5]
    sns.boxplot(x='Parking Availability', y='Exclusive Area', data=df[df['Parking Availability'].isin(top_cats)])
    plt.title("Area vs Parking")

    plt.subplot(2, 3, 5)
    corr = df.select_dtypes(include=np.number).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")

    plt.tight_layout()
    plt.savefig("eda_plots.png")
    plt.show()
    plt.close()

def train_test_split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def Calculate_Metrics(algorithm, y_pred, y_test, y_score=None):
    le_target = joblib.load(os.path.join(MODEL_DIR, 'target_label_encoder1.pkl'))
    categories = le_target.classes_

    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, average='macro') * 100
    rec = recall_score(y_test, y_pred, average='macro') * 100
    f1 = f1_score(y_test, y_pred, average='macro') * 100

    summary = (
        f"\n {algorithm} Evaluation:\n"
        f"Accuracy: {acc:.2f}%\n"
        f"Precision: {prec:.2f}%\n"
        f"Recall: {rec:.2f}%\n"
        f"F1-Score: {f1:.2f}%\n\n"
        f"classification_report: {classification_report(y_test, y_pred, labels=range(len(categories)), target_names=categories)}"
    )

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=categories, yticklabels=categories, cmap='Blues')
    plt.title(f'{algorithm} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'results/{algorithm.replace(" ", "_")}_confusion_matrix.png')
    plt.show()
    plt.close()

    if y_score is not None:
        y_test_bin = label_binarize(y_test, classes=range(len(categories)))
        plt.figure(figsize=(7, 6))

        if len(categories) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        else:
            for i in range(len(categories)):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{categories[i]} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f"{algorithm} - ROC Curve")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results/{algorithm.replace(" ", "_")}_roc_curve.png')
        plt.show()
        plt.close()

    return summary

def train_logistic_regression(X_train, y_train, X_test, y_test):
    path = os.path.join(MODEL_DIR, 'logistic_regression1.pkl')
    if os.path.exists(path):
        model = joblib.load(path)
    else:
        model = LogisticRegression()
        model.fit(X_train, y_train)
        joblib.dump(model, path)
    return Calculate_Metrics("Logistic Regression", model.predict(X_test), y_test, model.predict_proba(X_test))

def train_knn_classifier(X_train, y_train, X_test, y_test):
    path = os.path.join(MODEL_DIR, 'knn_classifier1.pkl')
    if os.path.exists(path):
        model = joblib.load(path)
    else:
        model = KNeighborsClassifier(n_neighbors=10)
        model.fit(X_train, y_train)
        joblib.dump(model, path)
    return Calculate_Metrics("KNN Classifier", model.predict(X_test), y_test, model.predict_proba(X_test))

def train_hybrid_extra_tree_ann(X_train, y_train, X_test, y_test, top_k=100):
    model_path = os.path.join(MODEL_DIR, 'hybrid_extra_tree_ann1.pkl')
    if os.path.exists(model_path):
        model, idx = joblib.load(model_path)
        X_test_reduced = X_test[:, idx]
    else:
        selector = ExtraTreesClassifier(n_estimators=100, random_state=42)
        selector.fit(X_train, y_train)
        importances = selector.feature_importances_
        idx = np.argsort(importances)[::-1][:top_k]
        X_train_reduced = X_train[:, idx]
        X_test_reduced = X_test[:, idx]
        model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
        model.fit(X_train_reduced, y_train)
        joblib.dump((model, idx), model_path)

    return Calculate_Metrics("Hybrid AI Model", model.predict(X_test_reduced), y_test, model.predict_proba(X_test_reduced))

def predict_class_only_real_estate(sample_df, model_name='logistic_regression1.pkl'):
    import os
    import joblib
    import scipy.sparse
    from scipy.sparse import hstack, csr_matrix

    model_path = os.path.join("models", model_name)
    loaded = joblib.load(model_path)

    _, X_sample, _ = preprocess_data(sample_df.copy(), is_train=False)

    # Convert to CSR format before any slicing
    X_sample = X_sample.tocsr()

    if model_name == 'hybrid_extra_tree_ann1.pkl':
        estimator, selected_indices = loaded

        # ✅ Fix: Pad if not enough features
        if X_sample.shape[1] <= max(selected_indices):
            missing = max(selected_indices) + 1 - X_sample.shape[1]
            X_sample = hstack([X_sample, csr_matrix((X_sample.shape[0], missing))]).tocsr()

        # ✅ Now safely slice
        X_sample = X_sample[:, selected_indices]
        y_pred = estimator.predict(X_sample)
    else:
        model = loaded
        y_pred = model.predict(X_sample)

    return y_pred

