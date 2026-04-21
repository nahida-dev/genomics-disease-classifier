# src/train.py

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

from src.config import *
from src.preprocessing import (
    load_data,
    normalize_data,
    extract_labels,
)


def train():
    file_path = RAW_DATA_DIR / "GSE15852_series_matrix.txt"

    # ------------------------
    # Load + preprocess
    # ------------------------
    print("Loading data...")
    df = load_data(file_path)
    df = normalize_data(df)

    # Transpose: samples as rows
    X = df.T
    y = np.array(extract_labels(file_path))

    print("Original shape:", X.shape)

    # ------------------------
    # Models
    # ------------------------
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE
        ),
    }

    # ------------------------
    # Cross-validation (BEFORE split, with feature selection inside pipeline)
    # ------------------------
    print("\nRunning cross-validation...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_summary = {}
    cv_results = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ("feature_selection", SelectKBest(score_func=f_classif, k=200)),
            ("model", model),
        ])

        cv_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="roc_auc"
        )

        cv_summary[name] = {
            "mean_roc_auc": cv_scores.mean(),
            "std_roc_auc": cv_scores.std(),
        }
        cv_results[name] = cv_scores

        print(f"\n{name} CV ROC-AUC scores: {cv_scores}")
        print(
            f"{name} Mean CV ROC-AUC: {cv_scores.mean():.4f} "
            f"(+/- {cv_scores.std():.4f})"
        )

    # ------------------------
    # Train/test split
    # ------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # ------------------------
    # Feature selection (fit ONLY on train)
    # ------------------------
    print("\nApplying feature selection...")

    selector = SelectKBest(score_func=f_classif, k=200)

    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    print("After feature selection:", X_train.shape)

    selected_indices = selector.get_support(indices=True)
    selected_feature_names = X.columns[selected_indices]
    selected_scores = selector.scores_[selected_indices]
    selected_features_df = pd.DataFrame(
        {
            "rank": range(1, len(selected_feature_names) + 1),
            "feature_name": selected_feature_names,
            "f_score": selected_scores,
        }
    ).sort_values("f_score", ascending=False, ignore_index=True)

    print("\nTop selected features (first 20):")
    for _, row in selected_features_df.head(20).iterrows():
        print(f"{row['feature_name']}: {row['f_score']:.4f}")

    best_model = None
    best_score = float("-inf")
    best_model_name = ""

    # ------------------------
    # Training loop
    # ------------------------
    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        roc = roc_auc_score(y_test, probs)

        print(f"{name} Accuracy: {acc:.4f}")
        print(f"{name} ROC-AUC: {roc:.4f}")
        print(classification_report(y_test, preds))

        cv_mean_roc = cv_summary[name]["mean_roc_auc"]

        # Select the model using cross-validation performance rather than
        # a single train/test split score.
        if cv_mean_roc > best_score:
            best_score = cv_mean_roc
            best_model = model
            best_model_name = name

    print("\nCross-validation summary:")
    for name in models:
        print(
            f"{name} Mean CV ROC-AUC: "
            f"{cv_summary[name]['mean_roc_auc']:.4f} "
            f"(+/- {cv_summary[name]['std_roc_auc']:.4f})"
        )

    # ------------------------
    # Save model + selector
    # ------------------------
    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(best_model, MODEL_DIR / "model.pkl")
    joblib.dump(selector, MODEL_DIR / "selector.pkl")
    selected_features_df.to_csv(MODEL_DIR / "selected_200_features.csv", index=False)

    print(f"\nBest model (selected by mean CV ROC-AUC): {best_model_name}")
    print(f"Best mean CV ROC-AUC: {best_score:.4f}")
    print("Model saved to models/model.pkl")
    print("Selector saved to models/selector.pkl")
    print("Selected features saved to models/selected_200_features.csv")


if __name__ == "__main__":
    train()
