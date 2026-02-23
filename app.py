import streamlit as st
import joblib
import os

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Lex Assist Mini", layout="centered")

st.title("⚖️ Lex Assist Mini")
st.markdown("### AI-Powered Legal Case Outcome Predictor")

st.write(
    "Enter a legal case description below and the model will predict the likely case outcome."
)

# ======================
# LOAD MODEL
# ======================
MODEL_PATH = "model/model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model not found. Please run train.py first.")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# ======================
# USER INPUT
# ======================
st.subheader("📝 Enter Case Text")

text = st.text_area(
    "Paste legal case text here:",
    height=220,
    placeholder="Example: The court considered the previous judgment and referred to..."
)

col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("🔍 Predict Outcome")

with col2:
    sample_btn = st.button("✨ Try Sample")

# ======================
# SAMPLE TEXT
# ======================
SAMPLE_TEXT = (
    "The court considered earlier precedents and referred to multiple prior judgments "
    "before delivering the final decision."
)

if sample_btn:
    text = SAMPLE_TEXT
    st.info("Sample text loaded. Click Predict Outcome.")

# ======================
# PREDICTION
# ======================
if predict_btn:
    if text.strip() == "":
        st.warning("⚠️ Please enter some legal text.")
    else:
        prediction = model.predict([text])[0]

        st.success("✅ Prediction Complete")
        st.markdown(f"### 🏷️ Predicted Outcome: **{prediction.upper()}**")

# ======================
# FOOTER
# ======================
st.divider()
st.caption("Built with ❤️ using NLP and Machine Learning")