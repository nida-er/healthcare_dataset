import streamlit as st
import numpy as np
import joblib
import os

# --- Model ve scaler yükleme ---
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, 'notebooks', 'model_mvp.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'notebooks', 'scaler.pkl')

@st.cache_resource
def load_model():
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model()

# --- Label encoding (notebook ile aynı sırada) ---
medical_conditions = sorted(['Arthritis', 'Asthma', 'Cancer', 'Diabetes', 'Hypertension', 'Obesity'])
condition_map = {cond: i for i, cond in enumerate(medical_conditions)}

# --- Arayüz ---
st.title('🏥 Fatura Tutarı Tahmini')
st.markdown('Hasta bilgilerini girerek tahmini fatura tutarını hesaplayın.')
st.divider()

col1, col2 = st.columns(2)

with col1:
    condition = st.selectbox('Hastalık (Medical Condition)', medical_conditions)

with col2:
    yatis_gun = st.number_input('Yatış Süresi (gün)', min_value=1, max_value=60, value=5)

st.divider()

if st.button('Tahmin Et', use_container_width=True, type='primary'):
    features = np.array([[
        condition_map[condition],
        yatis_gun
    ]])

    features_sc = scaler.transform(features)
    prediction  = model.predict(features_sc)[0]

    st.subheader('Tahmin Sonucu')
    st.metric(label='Tahmini Fatura Tutarı', value=f'${prediction:,.2f}')