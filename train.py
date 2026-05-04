import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.sparse import vstack

from data_preprocessing import preprocess
from tfidf_features import extract_features


# ✅ SAFE PATHS
MODEL_PATH = "models/pac_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
META_COLS_PATH = "models/meta_features.npy"
REPORT_PATH = "models/evaluation_report.txt"
CM_PLOT_PATH = "models/confusion_matrix.png"


def fix_label_column(df):
    """
    Ensure dataframe has 'fraudulent' column
    """

    if 'fraudulent' not in df.columns:
        if 'label' in df.columns:
            df.rename(columns={'label': 'fraudulent'}, inplace=True)
        else:
            raise Exception("Dataset must contain 'fraudulent' or 'label' column")

    return df


def add_irrelevant_class(df, X):

    print("[INFO] Generating 'irrelevant' class samples...")

    genuine_df = df[df['fraudulent'] == 0]

    n_irrelevant = max(100, int(len(genuine_df) * 0.05))

    idx = genuine_df.sample(n=n_irrelevant, random_state=42).index

    irr_rows = df.loc[idx].copy()
    irr_rows['fraudulent'] = 2

    df_aug = pd.concat([df, irr_rows], ignore_index=True)

    # convert sparse matrix
    X = X.tocsr()
    X_irr = X[idx]

    X_aug = vstack([X, X_irr])
    y_aug = df_aug['fraudulent'].values

    print(f"Genuine: {np.sum(y_aug==0)}")
    print(f"Fake: {np.sum(y_aug==1)}")
    print(f"Irrelevant: {np.sum(y_aug==2)}")

    return X_aug, y_aug


def train():

    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("[INFO] Preprocessing...")
    df = preprocess()

    # ✅ FIX COLUMN HERE
    df = fix_label_column(df)

    print("[INFO] Extracting TF-IDF features...")
    X, y, vectorizer, meta_cols = extract_features(df)

    X_aug, y_aug = add_irrelevant_class(df, X)

    print("[INFO] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_aug,
        y_aug,
        test_size=0.3,
        random_state=42,
        stratify=y_aug
    )

    print("[INFO] Training model...")
    model = PassiveAggressiveClassifier(
        C=0.1,
        max_iter=500,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("[INFO] Predicting...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc*100:.2f}%")

    cv_scores = cross_val_score(model, X_aug, y_aug, cv=5)
    print(f"Cross Validation Accuracy: {cv_scores.mean()*100:.2f}%")

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Genuine", "Fake", "Irrelevant"]
    )

    print(report)

    cm = confusion_matrix(y_test, y_pred)

    # Plot
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.savefig(CM_PLOT_PATH)

    # Save report
    with open(REPORT_PATH, "w") as f:
        f.write("Evaluation Report\n\n")
        f.write(f"Accuracy: {acc*100:.2f}%\n\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    # Save model
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    np.save(META_COLS_PATH, meta_cols)

    print("[INFO] Model saved successfully")

    return model, vectorizer, meta_cols, acc


if __name__ == "__main__":
    model, vectorizer, meta_cols, acc = train()
    print(f"\nTraining completed with accuracy: {acc*100:.2f}%")