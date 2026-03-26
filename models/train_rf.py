import datetime as dt
import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
DATA_DIR = Path(os.getenv("DATA_DIR"))
TRAIN_DATA_PATH = DATA_DIR / "04_multi_target" / "ess_model_ready_v2_train.csv"
TEST_DATA_PATH = DATA_DIR / "04_multi_target" / "ess_model_ready_v2_test.csv"
MODELS_DIR = Path(os.getenv("MODELS_DIR"))


def train_one_target(
    df_train: pd.DataFrame, df_test: pd.DataFrame, target_col: str, out_dir: Path
) -> None:
    """
    Train a Random Forest classifier for a single target column and save the model.
    """

    outcome_cols = ["hltprhb", "hltprhc", "hltprdi"]
    if target_col not in outcome_cols:
        raise ValueError(
            f"Unknown target '{target_col}'. Expected one of: {outcome_cols}"
        )

    feature_cols = [c for c in df_train.columns if c not in outcome_cols]

    X = df_train[feature_cols].copy()
    y = df_train[target_col].copy()
    X_test = df_test[feature_cols].copy()
    y_test = df_test[target_col].copy()

    uniq = sorted(y.dropna().unique())
    print("\n==============================")
    print(f"Training target: {target_col}")
    print(f"Unique values in target: {uniq}")
    print("Target distribution (normalized):")
    print(y.value_counts(normalize=True))
    print("==============================\n")

    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=5, random_state=42, n_jobs=-1
    )

    rf_model.fit(X, y)

    y_pred = rf_model.predict(X_test)

    roc_auc = None
    if hasattr(rf_model, "predict_proba") and len(rf_model.classes_) == 2:
        y_prob = rf_model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)

    accuracy = accuracy_score(y_test, y_pred)

    print("Model performance on test set:")
    print(f"Accuracy: {accuracy:.3f}")
    if roc_auc is not None:
        print(f"ROC-AUC:  {roc_auc:.3f}")
    else:
        print("ROC-AUC:  not computed (target is not binary in the expected way)")
    print()

    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print()

    out_dir.mkdir(parents=True, exist_ok=True)

    today = dt.datetime.today().strftime("%Y-%m-%d")
    model_path = out_dir / f"rf_{target_col}_{today}.pkl"

    joblib.dump(rf_model, model_path)
    print(f"Saved trained model to: {model_path}")


def main():
    train_data_path = TRAIN_DATA_PATH
    test_data_path = TEST_DATA_PATH
    print("\npath to data:", train_data_path)
    print("data-path is file:", train_data_path.is_file(), "\n")
    models_dir = MODELS_DIR

    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)
    print("Loaded train dataset:", train_data_path)
    print("Shape:", df_train.shape)
    print("Loaded train dataset:", test_data_path)
    print("Shape:", df_test.shape)

    targets = ["hltprhb", "hltprhc", "hltprdi"]
    for t in targets:
        train_one_target(df_train, df_test, t, models_dir)

    print("\n Training completed for all targets.")


if __name__ == "__main__":
    main()
