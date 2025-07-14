# interface.py

import io
import streamlit as st
import time
import requests
import pandas as pd
from scripts.utils import load_data
from scripts.make_graphs import plot_dataset, plot_suggestions, plot_convergence

ST_API_URL = "http://localhost:8000"

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf.getvalue()

st.title("Bayesian Optimization Web Interface")

# Inputs
func = st.selectbox("Select function", ["FE", "FD", "FM"])
data_filename = st.text_input(
    "Data filename (in datasets/synth_data, dims available: 2, 5, 10, 20)",
    "FE_d2_N50.json"
)
features = st.text_input("Feature columns (comma-separated)", "x1,x2").split(",")
target = st.text_input("Target column", "y")
n_iter = st.number_input("Number of BO iterations", min_value=1, value=30)
seed = st.number_input("Random seed", min_value=0, value=23)
acq = st.selectbox("Acquisition function", ["Expected Improvement", "Uniform Confidence Band", "Thompson Sampling"])
surrogate = st.selectbox("Surrogate model", ["Gaussian Process", "Bayesian Linear Regression", "K-Nearest Neighbors", "Random Forest"])
opt_method = st.selectbox("Optimization method", ["Gradient Optimization", "Random Sampling"])

if st.button("Run Optimization"):
    payload = {
        "func": func,
        "data_filename": data_filename,
        "feature_columns": [f.strip() for f in features],
        "target_column": target,
        "n_iter": n_iter,
        "seed": seed,
        "acquisition": acq,
        "surrogate_kind": surrogate,
        "opt_method": opt_method,
    }
    resp = requests.post(f"{ST_API_URL}/optimize", json=payload)
    if resp.status_code != 200:
        st.error(f"Error: {resp.text}")
    else:
        job_id = resp.json()["job_id"]
        st.success(f"Job submitted with ID: {job_id}")

        status_placeholder = st.empty()
        while True:
            status_resp = requests.get(f"{ST_API_URL}/jobs/{job_id}")
            status = status_resp.json().get("status")
            status_placeholder.info(f"Status: {status}")
            if status == "finished":
                files = status_resp.json().get("files", [])
                task_folder = f"{func}_d{len(features)}_{job_id}"

                if "bo_preds.json" in files:
                    # 1) Load predictions
                    json_url = f"{ST_API_URL}/results/{task_folder}/bo_preds.json"
                    r = requests.get(json_url)
                    data = r.json()
                    preds = pd.DataFrame(data)

                    st.subheader("Predictions")
                    st.dataframe(preds)
                    st.download_button("Download predictions JSON", r.text, file_name="bo_preds.json", mime="application/json")

                    # 2) Load original dataset
                    data_path = f"datasets/synth_data/{data_filename}"
                    _, _, df = load_data(data_path, features, target)

                    # 3) Suggestions plot
                    best_x = preds[features[0]].tolist()
                    best_y = preds["pred_y"].tolist()
                    fig2 = plot_suggestions(df, features[0], preds["global_max"].iloc[0], best_x, best_y, target, func, len(features), out_dir=None)  # ← EDIT
                    st.pyplot(fig2)
                    st.download_button("Download suggestions plot", fig_to_bytes(fig2), file_name="suggestions.png", mime="image/png")

                    # 4) Convergence plot
                    fig3 = plot_convergence(preds["pred_y"].tolist(), preds["train_max"].iloc[0], preds["global_max"].iloc[0], func, out_dir=None)  # ← EDIT
                    st.pyplot(fig3)
                    st.download_button("Download convergence plot", fig_to_bytes(fig3), file_name="convergence.png", mime="image/png")

                break
            elif status == "failed":
                st.error("Job failed. Check logs on server.")
                break
            time.sleep(2)
