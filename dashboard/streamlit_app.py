import sys
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import streamlit as st

from src.config import MODEL_DIR, RAW_DATA_DIR
from src.preprocessing import extract_labels, load_data, normalize_data


st.set_page_config(
    page_title="Genomics Disease Classifier",
    layout="wide",
)


def format_label(value: int) -> str:
    return "Cancer" if value == 1 else "Normal"


def api_predict(api_base_url: str, features: list[float]):
    payload = json.dumps({"features": features}).encode("utf-8")
    request = Request(
        f"{api_base_url.rstrip('/')}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            "Could not reach the API. Make sure FastAPI is running and the URL is correct."
        ) from exc


@st.cache_resource
def load_artifacts():
    selector = joblib.load(MODEL_DIR / "selector.pkl")

    importance_df = None
    importance_path = MODEL_DIR / "top10_genes.csv"
    if importance_path.exists():
        importance_df = pd.read_csv(importance_path)

    return selector, importance_df


@st.cache_data
def load_reference_data():
    file_path = RAW_DATA_DIR / "GSE15852_series_matrix.txt"
    df = normalize_data(load_data(file_path))
    labels = extract_labels(file_path)

    samples = df.T.reset_index(drop=True)
    metadata = pd.DataFrame(
        {
            "sample_index": range(len(samples)),
            "known_label": [format_label(label) for label in labels[: len(samples)]],
        }
    )

    return samples, metadata


def render_result(prediction: int, probability: float, source_name: str):
    label = format_label(prediction)

    st.subheader("Prediction Result")
    st.caption(f"Source: {source_name}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Class", label)
    col2.metric("Cancer Probability", f"{probability:.2%}")
    col3.metric("Normal Probability", f"{1 - probability:.2%}")

    if prediction == 1:
        st.error("The model classified this sample as Cancer.")
    else:
        st.success("The model classified this sample as Normal.")


def validate_uploaded_sample(user_df: pd.DataFrame, expected_features: int):
    if user_df.shape[0] != 1:
        return "CSV must contain exactly one row."

    if user_df.shape[1] != expected_features:
        return (
            f"CSV must contain exactly {expected_features} feature columns. "
            f"Received {user_df.shape[1]}."
        )

    if user_df.isnull().any().any():
        return "CSV contains missing values. Please provide a complete row."

    return None


selector, importance_df = load_artifacts()
reference_samples, reference_meta = load_reference_data()
expected_feature_count = int(selector.n_features_in_)

st.title("Genomics Disease Classifier")
st.write(
    "Use a reference sample from the dataset or upload a single processed sample "
    "to predict whether the expression profile looks more like Cancer or Normal."
)

info1, info2, info3 = st.columns(3)
info1.metric("Reference Samples", len(reference_samples))
info2.metric("Expected Features", expected_feature_count)
info3.metric("Model Output", "Cancer vs Normal")

st.sidebar.header("Prediction Input")
api_base_url = st.sidebar.text_input(
    "API base URL",
    value=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"),
    help="The Streamlit app sends prediction requests to this FastAPI server.",
)
input_mode = st.sidebar.radio(
    "Choose input method",
    ["Reference sample", "Upload CSV"],
)

if input_mode == "Reference sample":
    sample_idx = st.sidebar.number_input(
        "Sample index",
        min_value=0,
        max_value=len(reference_samples) - 1,
        value=0,
        step=1,
    )

    selected_meta = reference_meta.iloc[int(sample_idx)]
    st.sidebar.caption(
        f"Known dataset label: {selected_meta['known_label']}"
    )

    if st.sidebar.button("Run Prediction", use_container_width=True):
        sample = reference_samples.iloc[[int(sample_idx)]]
        try:
            result = api_predict(
                api_base_url=api_base_url,
                features=sample.iloc[0].astype(float).tolist(),
            )

            if "error" in result:
                st.error(result["error"])
            else:
                render_result(
                    prediction=int(result["prediction"]),
                    probability=float(result["cancer_probability"]),
                    source_name=f"Reference sample #{int(sample_idx)}",
                )

                st.subheader("Selected Sample Preview")
                st.dataframe(sample.iloc[:, :10], use_container_width=True)

                st.caption(
                    "Preview shows the first 10 features only. The model uses the full "
                    f"{expected_feature_count}-feature vector."
                )
        except Exception as exc:
            st.error(str(exc))
    else:
        st.subheader("Reference Sample Preview")
        preview_col1, preview_col2 = st.columns([2, 1])
        preview_col1.dataframe(
            reference_samples.iloc[[int(sample_idx)]].iloc[:, :10],
            use_container_width=True,
        )
        preview_col2.dataframe(
            reference_meta.iloc[[int(sample_idx)]],
            use_container_width=True,
        )
        st.caption("Click Run Prediction in the sidebar to score this sample.")

else:
    st.subheader("Upload a Custom Sample")
    st.write(
        "Upload a CSV containing exactly one row and "
        f"{expected_feature_count} numeric feature columns."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            validation_error = validate_uploaded_sample(
                user_df, expected_feature_count
            )

            if validation_error:
                st.error(validation_error)
            else:
                try:
                    result = api_predict(
                        api_base_url=api_base_url,
                        features=user_df.iloc[0].astype(float).tolist(),
                    )

                    if "error" in result:
                        st.error(result["error"])
                    else:
                        render_result(
                            prediction=int(result["prediction"]),
                            probability=float(result["cancer_probability"]),
                            source_name=uploaded_file.name,
                        )

                        st.subheader("Uploaded Sample Preview")
                        st.dataframe(user_df.iloc[:, :10], use_container_width=True)
                        st.caption("Preview shows the first 10 uploaded features.")
                except Exception as exc:
                    st.error(str(exc))

        except Exception as exc:
            st.error(f"Error processing file: {exc}")
    else:
        st.info(
            "Tip: if you do not have a custom sample yet, use the Reference sample "
            "mode in the sidebar to explore the model."
        )

if importance_df is not None:
    st.subheader("Top Important Genes")
    chart_df = importance_df.head(10).copy().sort_values("importance")

    chart_col1, chart_col2 = st.columns([1, 1])
    chart_col1.dataframe(importance_df, use_container_width=True)
    chart_col2.bar_chart(chart_df.set_index("gene")["importance"])

st.subheader("How To Use This App")
st.write(
    """
    1. Start the FastAPI server and confirm the API base URL in the sidebar.
    2. Choose **Reference sample** to test the model immediately with built-in data.
    3. Choose **Upload CSV** if you already have one processed sample with the full feature set.
    4. Click **Run Prediction** or upload your file to send the request through the API.
    """
)

st.subheader("About")
st.write(
    """
    This dashboard presents the trained genomics classifier in a user-friendly way.

    - Uses the GEO microarray dataset `GSE15852`
    - Applies feature selection before prediction
    - Classifies samples as Cancer or Normal
    - Surfaces the most important genes from the trained model
    """
)
