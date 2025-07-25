import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Set up Streamlit
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.title("🧪 Diabetes Risk Predictor")
st.markdown(
     """
    <div style='text-align: left; font-size: 16px;'>
        <a href="https://www.linkedin.com/in/abdullah-owais-23750920b" target="_blank">
            <img src="https://img.shields.io/badge/-LinkedIn-blue?logo=linkedin&style=flat-square">
        </a>
        <a href="https://github.com/abd-04" target="_blank">
            <img src="https://img.shields.io/badge/-GitHub-black?logo=github&style=flat-square">
        </a>
    </div>
    """,
    unsafe_allow_html=True
)


st.write("Logistic regression to help predict the risk of diabetes based on 3 imp factors ")




# Load Data & Model
df = pd.read_csv("diabetes.csv")
params = joblib.load("logistic_model.pkl")
w = params["weights"]
b = params["bias"]
scaler = params["scaler"]
losses = params.get("losses", None)

# Sigmoid & Predict
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, w, b):
    z = np.dot(X, w) + b
    prob = sigmoid(z)
    y_pred = (prob >= 0.5).astype(int)
    return y_pred[0][0], prob[0][0]

# 🚻 Input Fields
st.subheader("👤 Enter Your Medical Information")
glucose = st.slider("Glucose Level (mg/dL)", 0, 200, 120)
bmi = st.slider("Body Mass Index (BMI)", 0.0, 70.0, 25.0)
age = st.slider("Age (years)", 10, 100, 30)

# 🚀 Predict Button
if st.button("🔍 Predict Risk"):
    user_input = np.array([[glucose, bmi, age]])
    user_input_scaled = scaler.transform(user_input)

    y_pred, prob = predict(user_input_scaled, w, b)
    prob = round(prob, 2)

    if prob < 0.31:
        st.success(f"✅ Low Risk (Probability: {prob}) – You're likely safe.")
    elif prob < 0.61:
        st.warning(f"⚠️ Moderate Risk (Probability: {prob}) – Consider regular checkups.")
    else:
        st.error(f"🚨 High Risk (Probability: {prob}) – Please consult a doctor.")



# Real-time probability visualization
st.header("📈 Probability Curve against Glucose Levels")

# Use BMI and Age from user, sweep Glucose
bmi_fixed = bmi
age_fixed = age
glucose_range = np.linspace(0, 200, 300)
input_grid = np.array([[g, bmi_fixed, age_fixed] for g in glucose_range])
input_grid_scaled = scaler.transform(input_grid)
z_vals = np.dot(input_grid_scaled, w) + b
prob_vals = sigmoid(z_vals)

# Predict for actual user input
user_scaled = scaler.transform(np.array([[glucose, bmi_fixed, age_fixed]]))
user_prob = sigmoid(np.dot(user_scaled, w) + b)[0][0]

# Plot
fig2, ax2 = plt.subplots()
ax2.plot(glucose_range, prob_vals, color='blue', label="Sigmoid Probability Curve")
ax2.axhline(0.5, color='gray', linestyle='--', linewidth=1)
ax2.axvline(glucose, color='red', linestyle='--', label=f"Your Glucose: {glucose}")
ax2.scatter(glucose, user_prob, color='red', s=50, zorder=5)
ax2.set_xlabel("Glucose Level (mg/dL)")
ax2.set_ylabel("Probability of Diabetes")
ax2.set_title("Sigmoid Curve with Your Input")
ax2.legend()
st.pyplot(fig2)


# 📉 Training Loss Curve
st.header("📉 Training Loss Curve")
if losses:
    fig1, ax1 = plt.subplots()
    ax1.plot(losses, label="Loss per Epoch", color='darkorange')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve During Training")
    ax1.legend()
    st.pyplot(fig1)
else:
    st.info("Loss history not found in the saved model.")




# Footer
st.caption("An effort by Abdullah Owais 💻🚀 | ML from Scratch ")

