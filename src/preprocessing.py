# src/preprocessing.py
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif


def _classify_sample_title(sample_title: str) -> int:
    normalized_title = sample_title.strip().strip('"').casefold()

    if normalized_title.startswith("normal "):
        return 0
    if normalized_title.startswith("cancer "):
        return 1

    raise ValueError(
        f"Unrecognized sample title format: {sample_title!r}. "
        "Expected titles starting with 'Normal ' or 'Cancer '."
    )


def extract_labels(file_path: str):
    labels = None

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("!Sample_title"):
                parts = line.strip().split("\t")[1:]
                labels = [_classify_sample_title(part) for part in parts]
                break

    if labels is None:
        raise ValueError(
            "Could not find '!Sample_title' metadata row in the expression file."
        )

    if not labels:
        raise ValueError("No sample labels were extracted from the expression file.")

    return labels

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep="\t", comment="!")

    # First column = gene IDs
    df = df.set_index(df.columns[0])

    # Convert to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop bad rows
    df = df.dropna()

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Set first column as index (gene names)
    df = df.set_index(df.columns[0])

    # Convert everything to numeric (force errors to NaN)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop rows with NaNs
    df = df.dropna()

    return df


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.astype(float)

    invalid_mask = numeric_df < -1
    if invalid_mask.to_numpy().any():
        invalid_count = int(invalid_mask.sum().sum())
        raise ValueError(
            "normalize_data received values below -1, which are invalid for "
            f"log1p. Found {invalid_count} invalid measurements."
        )

    normalized_df = np.log1p(numeric_df)

    if not np.isfinite(normalized_df.to_numpy()).all():
        raise ValueError("normalize_data produced non-finite values.")

    return normalized_df


# Approach 1: Variance-Based Feature Selection
def select_top_genes(df: pd.DataFrame, n=500):
    """
    Select top N most variable genes
    """
    # Variance across samples
    variances = df.var(axis=1)

    # Select top genes
    top_genes = variances.sort_values(ascending=False).head(n).index

    return df.loc[top_genes]

# Approach 2: Add Statistical Feature Selection
def statistical_selection(X, y, k=200):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new, selector


def load_gene_mapping(mapping_file):
    mapping = pd.read_csv(mapping_file, sep="\t", comment="#", low_memory=False)

    # Keep only relevant columns
    mapping = mapping[["ID", "Gene Symbol"]]

    # Drop missing gene symbols
    mapping = mapping.dropna()

    return dict(zip(mapping["ID"], mapping["Gene Symbol"]))
