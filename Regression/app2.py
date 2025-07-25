import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = joblib.load("models/xgboost.pkl")

# Page setup
st.set_page_config(page_title="💎 AI Health Predictor", layout="wide")

# 🔥 Custom CSS
st.markdown("""
<style>
.stApp {
    background-image: url('https://images.unsplash.com/photo-1579154203451-b3f3770c4b52?auto=format&fit=crop&w=2100&q=80');
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    font-family: 'Segoe UI', sans-serif;
}
.card {
    background: rgba(255, 255, 255, 0.08);
    padding: 30px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    color: white;
    margin-bottom: 20px;
}
.tip-card {
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 30px;
    margin-top: 20px;
    color: white;
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
}
.tip {
    font-size: 18px;
    margin: 10px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
div.stButton > button {
    background: linear-gradient(135deg, #00f2fe, #4facfe);
    color: white;
    font-size: 18px;
    font-weight: bold;
    padding: 12px 30px;
    border-radius: 10px;
    border: none;
    box-shadow: 0 6px 15px rgba(0, 242, 254, 0.3);
    transition: 0.3s ease-in-out;
}
div.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 12px 25px rgba(0, 242, 254, 0.4);
}
.sidebar-box {
    background: rgba(255, 255, 255, 0.15);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    color: white;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# Page header
st.markdown("<h1 style='text-align:center; color:white;'>💉 AI Diabetes Prediction Portal</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:white;'>A futuristic health predictor powered by machine learning</h4><br>", unsafe_allow_html=True)

# Input card in main area
st.markdown('<div class="card">', unsafe_allow_html=True)

cols = st.columns(4)
Pregnancies = cols[0].number_input("Pregnancies", 0, 20, 1)
Glucose = cols[1].slider("Glucose Level", 0, 200, 120)
BloodPressure = cols[2].slider("Blood Pressure", 0, 140, 70)
SkinThickness = cols[3].slider("Skin Thickness", 0, 100, 20)

cols2 = st.columns(4)
Insulin = cols2[0].slider("Insulin", 0, 900, 80)
BMI = cols2[1].slider("BMI", 0.0, 70.0, 25.5)
DiabetesPedigreeFunction = cols2[2].slider("Pedigree Function", 0.0, 3.0, 0.5)
Age = cols2[3].slider("Age", 1, 100, 30)

st.markdown('</div>', unsafe_allow_html=True)

# Predict button triggers sidebar output
if st.button("🚀 Predict My Outcome"):
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DiabetesPedigreeFunction, Age]])
    try:
        outcome = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]

        # Sidebar result & chart
        st.sidebar.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
        if outcome == 1:
            st.sidebar.markdown("### ⚠ You are likely *Diabetic*")
            st.sidebar.markdown("🩺 Please consult your doctor immediately.")
        else:
            st.sidebar.markdown("### ✅ You are likely *Non-Diabetic*")
            st.sidebar.markdown("🎉 Keep up your healthy lifestyle!")

        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.pie([probs[1], probs[0]], labels=['Diabetic','Non-Diabetic'],
               colors=['#ff4d4d','#4caf50'], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.sidebar.pyplot(fig)
        st.sidebar.markdown('</div>', unsafe_allow_html=True)

        # Animated Tips section in main area
        st.markdown("""
        <div class="tip-card">
            <h3>💡 Health Tips</h3>
            <div class="tip">🥗 Eat fiber-rich foods like oats, fruits & veggies</div>
            <div class="tip">🏃‍♂ Stay active — at least 30 mins daily</div>
            <div class="tip">💧 Drink plenty of water</div>
            <div class="tip">🚭 Avoid smoking & alcohol</div>
            <div class="tip">📅 Get regular checkups</div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Footer
st.markdown("<hr><center style='color:white;'>✨ Built with ❤ using Streamlit + XGBoost</center>", unsafe_allow_html=True)