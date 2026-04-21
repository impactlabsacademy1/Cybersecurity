import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap

# -----------------------------
# Load model and features
# -----------------------------
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Create SHAP explainer
explainer = shap.Explainer(model)

st.title("🔐 AI Phishing Website Detector")

st.write("Enter website features to check if it is phishing or legitimate.")

# -----------------------------
# Input fields (based on features)
# -----------------------------
inputs = {}

for feature in feature_names:
    inputs[feature] = st.number_input(f"{feature}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([inputs])

# -----------------------------
# Predict
# -----------------------------
if st.button("Analyze Website"):

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    result = "Legitimate" if pred == 1 else "Phishing"

    st.subheader(f"Prediction: {result}")
    st.write(f"Confidence: {round(prob,2)}")

    # -----------------------------
    # SHAP Explanation
    # -----------------------------
    shap_values = explainer(input_df)
    vals = shap_values.values

    if len(vals.shape) == 2:
        shap_vals = vals[0]
    elif len(vals.shape) == 3:
        shap_vals = vals[0, :, 1]

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "value": input_df.iloc[0].values,
        "shap_value": shap_vals
    })

    shap_df["abs_shap"] = np.abs(shap_df["shap_value"])
    shap_df = shap_df.sort_values(by="abs_shap", ascending=False)

    st.subheader("🔍 Top Reasons:")

    for _, row in shap_df.head(3).iterrows():
        if pred == 1:
            if row["shap_value"] > 0:
                st.write(f"✅ {row['feature']} supports legitimacy")
            else:
                st.write(f"⚠️ {row['feature']} raises suspicion")
        else:
            if row["shap_value"] < 0:
                st.write(f"⚠️ {row['feature']} indicates phishing")
            else:
                st.write(f"✅ {row['feature']} indicates legitimacy")