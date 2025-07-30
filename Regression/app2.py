import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = joblib.load("models/xgboost.pkl")

# Page configuration
st.set_page_config(page_title="💎 AI Health Predictor", layout="wide")

# Dark mode CSS
st.markdown("""
<style>
body, .stApp {
    background-color: #121212;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}
div.stButton > button {
    background: linear-gradient(135deg, #00f2fe, #4facfe);
    color: white;
    font-size: 18px;
    font-weight: bold;
    padding: 10px 25px;
    border-radius: 10px;
    border: none;
}
div.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 18px rgba(0, 242, 254, 0.4);
}
.tip-card {
    background: rgba(255, 255, 255, 0.08);
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
}
.tip-card h3, .tip {
    color: #ffffff !important;
    font-weight: 600;
}
.tip {
    margin: 8px 0;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# Titles
st.markdown("<h1 style='text-align:center;'>💉  Diabetes Prediction Portal</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>A futuristic health predictor powered by machine learning</h4><br>", unsafe_allow_html=True)

# Session state setup
for key in ['predicted', 'diabetic', 'show_tips', 'probs']:
    if key not in st.session_state:
        st.session_state[key] = False if key != 'probs' else None

# Sliders
st.markdown('<div class="card">', unsafe_allow_html=True)
cols = st.columns(4)
Pregnancies = cols[0].slider("Pregnancies", 0, 20, 1)
Glucose = cols[1].slider("Glucose Level", 0, 200, 120)
BloodPressure = cols[2].slider("Blood Pressure", 0, 140, 70)
SkinThickness = cols[3].slider("Skin Thickness", 0, 100, 20)

cols2 = st.columns(4)
Insulin = cols2[0].slider("Insulin", 0, 900, 80)
BMI = cols2[1].slider("BMI", 0.0, 70.0, 25.5)
DiabetesPedigreeFunction = cols2[2].slider("Pedigree Function", 0.0, 3.0, 0.5)
Age = cols2[3].slider("Age", 1, 100, 30)
st.markdown('</div>', unsafe_allow_html=True)

# Prediction
if st.button(" Predict Result"):
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DiabetesPedigreeFunction, Age]])
    try:
        outcome = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]
        st.session_state.predicted = True
        st.session_state.diabetic = bool(outcome)
        st.session_state.probs = probs
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Results
if st.session_state.predicted:
    if st.session_state.diabetic:
        st.markdown("""
        <h3 style="
            color: #FF3C38;
            font-weight: 700;
            text-shadow: 0 0 8px #FF3C38;
            background-color: rgba(255, 60, 56, 0.15);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        ">
            ⚠ You are likely Diabetic
        </h3>
        """, unsafe_allow_html=True)
        st.markdown("🩺 Please consult your doctor immediately.")
    else:
        st.markdown("""
        <h3 style="
            color: #4CAF50;
            font-weight: 700;
            text-shadow: 0 0 8px #4CAF50;
            background-color: rgba(76, 175, 80, 0.1);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        ">
            ✅ You are likely Non-Diabetic
        </h3>
        """, unsafe_allow_html=True)

    # Feature importance pie chart
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_values = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness,
                             Insulin, BMI, DiabetesPedigreeFunction, Age])

    importance = model.feature_importances_
    weighted_importance = importance * input_values
    percent_contrib = 100 * weighted_importance / weighted_importance.sum()

    # 🎨 Updated pie chart colors
    pie_colors = ['#8e44ad', '#3498db', '#1abc9c', '#f1c40f',
                  '#e67e22', '#e74c3c', '#2ecc71', '#9b59b6']

    # Layout: left = pie + legend, right = prob bar
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        left_cols = st.columns([1, 1])
        with left_cols[0]:
            st.markdown("#### Feature Impact & Contribution")
            for name, percent, color in zip(feature_names, percent_contrib, pie_colors):
                st.markdown(f"""
                <div style='display:flex;align-items:center;margin-bottom:5px;'>
                    <div style='width:18px;height:18px;background-color:{color};border-radius:4px;margin-right:10px;'></div>
                    <span style='font-size:14px;color:white;'>{name} – <strong>{percent:.1f}%</strong></span>
                </div>""", unsafe_allow_html=True)
        with left_cols[1]:
            fig, ax = plt.subplots(figsize=(3, 3), facecolor='none')
            ax.pie(
                percent_contrib,
                labels=None,
                colors=pie_colors,
                startangle=135,
                wedgeprops={'edgecolor': 'white'}
            )
            ax.axis('equal')
            st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(3, 3), facecolor='none')
        bars = ax2.bar(
            ['Non-Diabetic', 'Diabetic'],
            [st.session_state.probs[0] * 100, st.session_state.probs[1] * 100],
            color=['#3498db', '#e74c3c']
        )
        for bar in bars:
            bar.set_alpha(0.8)
            bar.set_edgecolor('white')
            bar.set_linewidth(1)

        ax2.set_title("🔍 Prediction Probability", fontsize=11, color='white')
        ax2.set_ylabel("Probability (%)", fontsize=9, color='white')
        ax2.tick_params(axis='x', labelsize=9, colors='white')
        ax2.tick_params(axis='y', labelsize=9, colors='white')
        for spine in ax2.spines.values():
            spine.set_color('white')
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<hr><center>✨ Built with ❤ using Streamlit + XGBoost</center>", unsafe_allow_html=True)