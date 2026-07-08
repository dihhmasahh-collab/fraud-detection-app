import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

st.set_page_config(page_title="Fraud Detection App", page_icon="🛡️", layout="centered")

st.markdown("""
<style>
    .main {
        background-color: #FAFAFA;
    }
    .title-container {
        background: linear-gradient(90deg, #1D3557 0%, #457B9D 100%);
        padding: 2rem 1.5rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .title-container h1 {
        color: white !important;
        font-size: 2rem;
        margin-bottom: 0.3rem;
    }
    .title-container p {
        color: #E0E6ED;
        font-size: 0.95rem;
        margin: 0;
    }
    div[data-testid="stForm"] {
        background-color: white;
        padding: 1.8rem;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #EAECEF;
    }
    .stButton button, div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(90deg, #E63946 0%, #D62828 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        width: 100%;
        transition: 0.2s;
    }
    .stButton button:hover, div[data-testid="stFormSubmitButton"] button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
    .result-fraud {
        background-color: #FDECEA;
        border-left: 6px solid #D62828;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .result-normal {
        background-color: #E9F7EF;
        border-left: 6px solid #2A9D8F;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open("fraud_detection_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

CATEGORIES = ['entertainment', 'food_dining', 'gas_transport', 'grocery_net',
              'grocery_pos', 'health_fitness', 'home', 'kids_pets', 'misc_net',
              'misc_pos', 'personal_care', 'shopping_net', 'shopping_pos', 'travel']

STATES = ['AK','AL','AR','AZ','CA','CO','CT','DC','FL','GA','HI','IA','ID','IL',
          'IN','KS','KY','LA','MA','MD','ME','MI','MN','MO','MS','MT','NC','ND',
          'NE','NH','NJ','NM','NV','NY','OH','OK','OR','PA','RI','SC','SD','TN',
          'TX','UT','VA','VT','WA','WI','WV','WY']

st.markdown("""
<div class="title-container">
    <h1>🛡️ Deteksi Fraud Transaksi Kartu Kredit</h1>
    <p>Masukkan detail transaksi untuk memprediksi apakah transaksi tersebut <b>normal</b> atau <b>fraud</b> menggunakan model machine learning.</p>
</div>
""", unsafe_allow_html=True)

with st.form("fraud_form"):
    col1, col2 = st.columns(2)

    with col1:
        amt = st.number_input("💵 Jumlah Transaksi (amt, USD)", min_value=0.0, value=50.0, step=1.0)
        category = st.selectbox("🏬 Kategori Merchant", CATEGORIES)
        gender = st.selectbox("👤 Jenis Kelamin Nasabah", ["M", "F"])
        state = st.selectbox("📍 State Nasabah", STATES)
        city_pop = st.number_input("🏙️ Populasi Kota Nasabah", min_value=0, value=50000, step=100)

    with col2:
        trans_date = st.date_input("📅 Tanggal Transaksi", value=datetime.now())
        trans_time = st.time_input("⏰ Jam Transaksi", value=datetime.now().time())
        age = st.number_input("🎂 Usia Nasabah", min_value=15, max_value=100, value=35)
        distance_km = st.number_input("📏 Jarak Nasabah ke Merchant (km)", min_value=0.0, value=50.0, step=1.0)

    submitted = st.form_submit_button("🔍 Prediksi Transaksi")

if submitted:
    input_df = pd.DataFrame([{
        "amt": amt,
        "city_pop": city_pop,
        "trans_hour": trans_time.hour,
        "trans_day": trans_date.day,
        "trans_month": trans_date.month,
        "trans_dayofweek": trans_date.weekday(),
        "age": age,
        "distance_km": distance_km,
        "category": category,
        "gender": gender,
        "state": state,
    }])

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.markdown(f"""
        <div class="result-fraud">
            <h3>🚨 FRAUD TERDETEKSI</h3>
            <p>Probabilitas fraud: <b>{proba:.2%}</b></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-normal">
            <h3>✅ Transaksi Normal</h3>
            <p>Probabilitas fraud: <b>{proba:.2%}</b></p>
        </div>
        """, unsafe_allow_html=True)

    st.progress(min(float(proba), 1.0))

st.markdown("---")
st.caption("Model machine learning ini dilatih menggunakan dataset Credit Card Transactions Fraud Detection (Kaggle) untuk keperluan tugas kelompok.")
