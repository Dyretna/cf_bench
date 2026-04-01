import datetime as dt
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from aiwhatif_cf.config.config import DiceConfig


def main():
    # Targets you want to train
    targets = ["hltprhb", "hltprhc"]

    # RF model template (we clone it for each target)
    base_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    for target in targets:
        print(f"\n=== Training model for target: {target} ===")

        # Create config for this target
        config = DiceConfig(target=target)

        # Load data
        df_train, df_test = load_dataset(config.train_data_path, config.test_data_path)

        # Train model
        train_rf_model(df_train, df_test, target, base_rf, config.model_dir)

    print("\nTraining completed for all targets.")


def load_dataset(
    train_data_path: Path, test_data_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n", "==== Data ===")
    print("\npath to data:", train_data_path)
    print("data-path is file:", train_data_path.is_file(), "\n")

    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)

    print("Loaded train dataset:", train_data_path)
    print("Shape:", df_train.shape)
    print("Loaded test dataset:", test_data_path)
    print("Shape:", df_test.shape)

    return df_train, df_test


def train_rf_model(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target: str,
    rf_model: RandomForestClassifier,
    out_dir: Path,
) -> None:
    """
    Train a Random Forest classifier for a single target column and save the model.
    """

    feature_cols = [c for c in df_train.columns if c not in target]

    X = df_train[feature_cols].copy()
    y = df_train[target].copy()
    X_test = df_test[feature_cols].copy()
    y_test = df_test[target].copy()

    uniq = sorted(y.dropna().unique())
    print("\n==============================")
    print(f"Unique values in target: {uniq}")
    print("Target distribution (normalized):")
    print(y.value_counts(normalize=True))
    print("==============================\n")

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
    model_path = out_dir / f"rf_{target}_{today}.pkl"

    joblib.dump(rf_model, model_path)
    print(f"Saved trained model to: {model_path}")


if __name__ == "__main__":
    main()
