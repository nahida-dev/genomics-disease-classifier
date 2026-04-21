# src/explain.py

import joblib
import numpy as np
import pandas as pd
import shap

from src.config import *
from src.preprocessing import load_data, normalize_data


# ------------------------
# Load gene mapping
# ------------------------
def load_gene_mapping(mapping_file):
    # Read file manually to find where table starts
    with open(mapping_file, "r") as f:
        lines = f.readlines()

    # Find header line (starts with "ID\t")
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("ID\t"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find header line in mapping file")

    # Read from correct header
    mapping = pd.read_csv(
        mapping_file,
        sep="\t",
        skiprows=header_idx,
        engine="python"
    )

    print("Columns:", mapping.columns.tolist())

    # Clean columns
    mapping = mapping[["ID", "Gene Symbol"]].dropna()

    return dict(zip(mapping["ID"], mapping["Gene Symbol"]))


def explain():
    file_path = RAW_DATA_DIR / "GSE15852_series_matrix.txt"
    mapping_file = RAW_DATA_DIR / "GPL96-57554.txt"

    print("Loading data...")

    # ------------------------
    # Load data
    # ------------------------
    df = load_data(file_path)
    df = normalize_data(df)

    X = df.T
    gene_names = df.index  # probe IDs

    print("Data shape:", X.shape)

    # ------------------------
    # Load model + selector
    # ------------------------
    print("Loading model and selector...")

    model = joblib.load(MODEL_DIR / "model.pkl")
    selector = joblib.load(MODEL_DIR / "selector.pkl")

    # Apply feature selection
    X_selected = selector.transform(X)

    # Get selected gene indices
    selected_indices = selector.get_support(indices=True)
    selected_probes = gene_names[selected_indices]

    # ------------------------
    # Map probe → gene name
    # ------------------------
    print("Mapping probe IDs to gene symbols...")

    gene_map = load_gene_mapping(mapping_file)

    mapped_genes = [
        gene_map.get(probe, probe) for probe in selected_probes
    ]

    # ------------------------
    # SHAP explanation
    # ------------------------
    print("Running SHAP...")

    # Use TreeExplainer for RandomForest (faster)
    if hasattr(model, "estimators_"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, X_selected)

    shap_values = explainer(X_selected)

    # ------------------------
    # Summary plot
    # ------------------------
    print("Generating SHAP summary plot...")

    shap.summary_plot(
        shap_values,
        X_selected,
        feature_names=mapped_genes,
        max_display=20
    )

    # ------------------------
    # Feature importance table
    # ------------------------
    print("\nCalculating feature importance...")

    shap_mean = np.abs(shap_values.values).mean(axis=0)

    importance_df = pd.DataFrame({
        "probe_id": selected_probes,
        "gene": mapped_genes,
        "importance": shap_mean
    }).sort_values(by="importance", ascending=False)

    # ------------------------
    # Print top 10 clearly
    # ------------------------
    print("\n" + "="*50)
    print("TOP 10 IMPORTANT GENES")
    print("="*50)

    top10 = importance_df.head(10)

    for i, row in top10.iterrows():
        print(f"{row['gene']} ({row['probe_id']}) → {row['importance']:.4f}")

    # ------------------------
    # Save outputs
    # ------------------------
    MODEL_DIR.mkdir(exist_ok=True)

    importance_df.to_csv(MODEL_DIR / "gene_importance.csv", index=False)
    top10.to_csv(MODEL_DIR / "top10_genes.csv", index=False)

    print("\n Saved:")
    print(" - models/gene_importance.csv")
    print(" - models/top10_genes.csv")


if __name__ == "__main__":
    explain()
