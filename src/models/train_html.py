
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# Project paths
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# URL-only feature list
FEATURES = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
]

def main():
    print("Loading PhiUSIIL dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("Original label counts:")
    print(df["label"].value_counts(dropna=False))

    # Verify label meanings in the official dataset documentation.
    # PhiUSIIL: 1 = legitimate, 0 = phishing.
    df = df.drop_duplicates(subset=["URL"]).copy()

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    original_labels = pd.to_numeric(df["label"], errors="coerce")

    # Keep valid feature rows and binary labels only.
    valid = (
        X.notna().all(axis=1)
        & original_labels.isin([0, 1])
    )

    X = X.loc[valid]
    y = (1 - original_labels.loc[valid]).astype(int)
    # Now 1 = phishing, 0 = legitimate.

    print("Records after cleaning:", len(X))
    print("Phishing:", int(y.sum()))
    print("Legitimate:", int((y == 0).sum()))

    # 70% training, 15% validation, 15% testing
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50,
        random_state=42, stratify=y_temp
    )

    # Save train, validation, and test datasets
    SPLIT_DIR = ROOT / "data" / "splits"
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df["URL"] = df.loc[X_train.index, "URL"]
    train_df["label"] = y_train
    train_df.to_csv(SPLIT_DIR / "train.csv", index=False)

    val_df = X_val.copy()
    val_df["URL"] = df.loc[X_val.index, "URL"]
    val_df["label"] = y_val
    val_df.to_csv(SPLIT_DIR / "validation.csv", index=False)

    test_df = X_test.copy()
    test_df["URL"] = df.loc[X_test.index, "URL"]
    test_df["label"] = y_test
    test_df.to_csv(SPLIT_DIR / "test.csv", index=False)

    print("Saved train, validation, and test CSV files to:", SPLIT_DIR)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    print("Training URL-only model...")
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    print("Validation ROC-AUC:", roc_auc_score(y_val, val_prob))

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    print("\nFinal test results:")
    print(classification_report(
        y_test,
        test_pred,
        labels=[0, 1],
        target_names=["Legitimate", "Phishing"],
        zero_division=0
    ))
    print("Test ROC-AUC:", roc_auc_score(y_test, test_prob))

    # Save model and its required feature order
    joblib.dump(
        {"model": model, "features": FEATURES},
        MODEL_DIR / "url_model.joblib"
    )

    print("Model saved to models/url_model.joblib")

if __name__ == "__main__":
    main()
