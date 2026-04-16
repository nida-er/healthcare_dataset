import streamlit as st
import numpy as np
import joblib
import os

# Model ve scaler yükle
# app.py, notebooks/ klasöründeki .pkl dosyalarını referans alır
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, '..', 'notebooks', 'model_classification.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'notebooks', 'scaler.pkl')

@st.cache_resource
def load_model():
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model()

label_maps = {
    'Gender'             : {'Male': 1, 'Female': 0},
    'Blood Type'         : {'A+': 0, 'A-': 1, 'B+': 2, 'B-': 3, 'AB+': 4, 'AB-': 5, 'O+': 6, 'O-': 7},
    'Medical Condition'  : {'Arthritis': 0, 'Asthma': 1, 'Cancer': 2, 'Diabetes': 3, 'Hypertension': 4, 'Obesity': 5},
    'Insurance Provider' : {'Aetna': 0, 'Blue Cross': 1, 'Cigna': 2, 'Medicare': 3, 'UnitedHealthcare': 4},
    'Admission Type'     : {'Elective': 0, 'Emergency': 1, 'Urgent': 2},
    'Medication'         : {'Aspirin': 0, 'Ibuprofen': 1, 'Lipitor': 2, 'Paracetamol': 3, 'Penicillin': 4},
}
result_map = {0: '🟠 Inconclusive', 1: '🔴 Abnormal', 2: '🟢 Normal'}

# Sayfa ayarları
st.set_page_config(page_title='Hasta Test Sonucu Tahmini', page_icon='🏥', layout='centered')

st.title('🏥 Hasta Test Sonucu Tahmini')
st.markdown('Hasta bilgilerini girerek test sonucunu tahmin edin.')
st.divider()

col1, col2 = st.columns(2)

with col1:
    age            = st.number_input('Yaş', min_value=1, max_value=120, value=45)
    billing        = st.number_input('Fatura Tutarı (USD)', min_value=0.0, value=15000.0, step=100.0)
    yatis_gun      = st.number_input('Yatış Süresi (gün)', min_value=1, max_value=60, value=5)
    admission_ay   = st.selectbox('Başvuru Ayı', list(range(1, 13)))
    gender         = st.selectbox('Cinsiyet', list(label_maps['Gender'].keys()))
    blood_type     = st.selectbox('Kan Grubu', list(label_maps['Blood Type'].keys()))

with col2:
    condition      = st.selectbox('Hastalık', list(label_maps['Medical Condition'].keys()))
    insurance      = st.selectbox('Sigorta', list(label_maps['Insurance Provider'].keys()))
    admission_type = st.selectbox('Başvuru Tipi', list(label_maps['Admission Type'].keys()))
    medication     = st.selectbox('İlaç', list(label_maps['Medication'].keys()))

st.divider()

if st.button('Tahmin Et', use_container_width=True, type='primary'):
    features = np.array([[
        age,
        yatis_gun,
        billing,
        admission_ay,
        label_maps['Gender'][gender],
        label_maps['Blood Type'][blood_type],
        label_maps['Medical Condition'][condition],
        label_maps['Insurance Provider'][insurance],
        label_maps['Admission Type'][admission_type],
        label_maps['Medication'][medication],
    ]])

    features_sc = scaler.transform(features)
    prediction  = model.predict(features_sc)[0]
    proba       = model.predict_proba(features_sc)[0]

    st.subheader('Tahmin Sonucu')
    st.markdown(f'### {result_map[prediction]}')

    st.subheader('Olasılık Dağılımı')
    for i, label in result_map.items():
        st.progress(float(proba[i]), text=f'{label}: %{proba[i]*100:.1f}')