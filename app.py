import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Multiple Disease Prediction", layout="wide")

# =========================================
# Helper: Load Model
# =========================================
def load_model(filename):
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
        if isinstance(model_data, tuple):
            model, scaler = model_data
        else:
            model = model_data
            scaler = None
    return model, scaler

# =========================================
# Sidebar Navigation
# =========================================
st.sidebar.title("Disease Prediction Dashboard")

selected_model = st.sidebar.selectbox(
    "Choose a Model",
    ["Parkinson's", "Kidney Disease", "Indian Liver Patient"]
)

# =========================================
# 1️⃣ Parkinson's Disease Prediction
# =========================================
if selected_model == "Parkinson's":
    st.title("Parkinson's Disease Prediction")

    # Input fields
    spread1 = st.number_input("Spread1")
    PPE = st.number_input("PPE")
    DFA = st.number_input("DFA")
    RPDE = st.number_input("RPDE")
    NHR = st.number_input("NHR")

    if st.button("Predict Parkinson's"):
        model, scaler = load_model("rfc.score.pkl")

        input_data = np.array([[spread1, PPE, DFA, RPDE, NHR]])

        if scaler:
            input_scaled = scaler.transform(input_data)
        else:
            input_scaled = input_data

        prediction = model.predict(input_scaled)[0]

        st.success("Prediction: {}".format(
            "Parkinson's Detected" if prediction == 1 else "No Parkinson's"
        ))

# =========================================
# 2️⃣ Chronic Kidney Disease Prediction
#     Using Main Features Only
# =========================================

elif selected_model == "Kidney Disease":
    st.title("Chronic Kidney Disease Prediction (Main Features Only)")

    # Main Inputs
    age = st.number_input("Age", 1, 120, 45)
    bp = st.number_input("Blood Pressure", 50, 200, 80)
    sg = st.number_input("Specific Gravity", 1.000, 1.030, 1.015)
    al = st.number_input("Albumin", 0, 5, 1)
    su = st.number_input("Sugar", 0, 5, 0)
    bgr = st.number_input("Blood Glucose Random", 50, 500, 120)
    bu = st.number_input("Blood Urea", 1, 300, 40)
    sc = st.number_input("Serum Creatinine", 0.1, 20.0, 1.2)
    hemo = st.number_input("Hemoglobin", 3.0, 20.0, 13.5)
    pcv = st.number_input("Packed Cell Volume", 20, 55, 40)

    # Default categorical values from dataset
    default_cat = {
        "rbc": 1,
        "pc": 1,
        "pcc": 0,
        "ba": 0,
        "htn": 0,
        "dm": 0,
        "cad": 0,
        "appet": 1,
        "pe": 0,
        "ane": 0
    }

    # Vector Builder for 22 Features
    def build_ckd_vector():
        numeric_main = [age, bp, sg, al, su, bgr, bu, sc,
                        hemo, pcv, 0, 0, 0]  # remaining numeric placeholders

        numeric_filled = [0 if (v is None or np.isnan(v)) else v for v in numeric_main]

        cat_encoded = [
            default_cat["rbc"], default_cat["pc"], default_cat["pcc"], default_cat["ba"],
            default_cat["htn"], default_cat["dm"], default_cat["cad"], default_cat["appet"],
            default_cat["pe"], default_cat["ane"]
        ]

        final = numeric_filled + cat_encoded
        return np.array(final).reshape(1, -1)

    if st.button("Predict CKD"):
        model, scaler = load_model("best_ckd_model.pkl")

        X = build_ckd_vector()
        prediction = model.predict(X)[0]

        if prediction == 1:
            st.error("⚠️ Chronic Kidney Disease Detected")
        else:
            st.success("✅ No CKD Detected")


# =========================================
# 3️⃣ Indian Liver Patient Prediction
# =========================================

elif selected_model == "Indian Liver Patient":
    st.title("Indian Liver Patient Prediction")

    age = st.number_input("Age", 0, 120, 45)
    total_bilirubin = st.number_input("Total Bilirubin")
    direct_bilirubin = st.number_input("Direct Bilirubin")
    alkaline_phosphotase = st.number_input("Alkaline Phosphotase")
    alamine_aminotransferase = st.number_input("Alamine Aminotransferase")
    aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase")
    total_proteins = st.number_input("Total Proteins")
    albumin = st.number_input("Albumin")
    ag_ratio = st.number_input("Albumin and Globulin Ratio")

    threshold = st.slider("Set Prediction Threshold", 0.50, 1.00, 0.70, 0.05)

    if st.button("Predict Liver Disease"):
        model, scaler = load_model("liver_disease_model.pkl")

        input_data = np.array([[
            age,
            total_bilirubin,
            direct_bilirubin,
            alkaline_phosphotase,
            alamine_aminotransferase,
            aspartate_aminotransferase,
            total_proteins,
            albumin,
            ag_ratio
        ]])

        input_scaled = scaler.transform(input_data)

        prob = model.predict_proba(input_scaled)[0]
        prob_no = prob[0]
        prob_yes = prob[1]

        st.subheader("Prediction Probabilities")
        st.info(f"No Liver Disease: {prob_no:.2f}")
        st.warning(f"Liver Disease: {prob_yes:.2f}")

        if prob_yes >= threshold:
            st.error(f"🚨 Liver Disease Detected (Confidence: {prob_yes:.2f})")
        else:
            st.success(f"🎉 No Liver Disease Detected (Confidence: {prob_no:.2f})")
