import time
import requests
import pandas as pd
import streamlit as st

API_BASE = st.sidebar.text_input("API Base URL", "http://127.0.0.1:8000")

st.title("Insurance Claims Fraud Scoring Demo")

# ---- Model Info ----
st.sidebar.subheader("Model Info")
if st.sidebar.button("Refresh /model/info"):
    try:
        info = requests.get(f"{API_BASE}/model/info", timeout=5).json()
        st.sidebar.json(info)
    except Exception as e:
        st.sidebar.error(f"Failed to fetch model info: {e}")

# ---- Input Form ----
st.subheader("Enter Claim Details")

with st.form("claim_form"):
    claim_amount = st.number_input("Claim Amount", min_value=0.01, value=15000.0, step=100.0)
    num_prior_claims = st.number_input("Number of Prior Claims", min_value=0, value=2, step=1)
    days_since_policy_start = st.number_input("Days Since Policy Start", min_value=0, value=120, step=1)
    vin = st.text_input("VIN (optional)", value="")

    submitted = st.form_submit_button("Score Fraud Risk")

if "history" not in st.session_state:
    st.session_state.history = []

# ---- Call API ----
if submitted:
    payload = {
        "claim_amount": float(claim_amount),
        "num_prior_claims": int(num_prior_claims),
        "days_since_policy_start": int(days_since_policy_start),
    }
    if vin.strip():
        payload["vin"] = vin.strip()

    try:
        t0 = time.time()
        r = requests.post(f"{API_BASE}/predict/fraud", json=payload, timeout=10)
        latency_ms = (time.time() - t0) * 1000.0

        if r.status_code != 200:
            st.error(f"API Error {r.status_code}: {r.text}")
        else:
            data = r.json()
            st.success(f"Scored in {latency_ms:.1f} ms")

            st.metric("Fraud Probability", f'{data["fraud_probability"]:.3f}')
            st.write(f'**Label:** {data["label"]}  |  **Threshold:** {data["threshold"]}')
            st.write(f'**VIN Status:** {data["vin_status"]}')
            st.write(f'**Degraded:** {data["degraded"]}  |  **Confidence:** {data["confidence"]}')

            st.subheader("Guardrails")
            st.write("**Flags**")
            st.json(data.get("guardrail_flags", {}))

            st.write("**Reasons**")
            reasons = data.get("guardrail_reasons", [])
            if reasons:
                for x in reasons:
                    st.warning(x)
            else:
                st.info("No guardrail reasons triggered.")

            with st.expander("Show features used"):
                st.json(data.get("features_used", {}))

            row = {
                "timestamp": time.strftime("%H:%M:%S"),
                "claim_amount": payload["claim_amount"],
                "num_prior_claims": payload["num_prior_claims"],
                "days_since_policy_start": payload["days_since_policy_start"],
                "vin": payload.get("vin", ""),
                "fraud_probability": data["fraud_probability"],
                "label": data["label"],
                "degraded": data["degraded"],
                "confidence": data["confidence"],
                "vin_status": data["vin_status"],
                "latency_ms": round(latency_ms, 1),
            }
            st.session_state.history.insert(0, row)
            st.session_state.history = st.session_state.history[:10]

    except Exception as e:
        st.error(f"Request failed: {e}")

# ---- History ----
st.subheader("Last Predictions (local)")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
else:
    st.info("No predictions yet. Submit the form above.")
