import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

st.set_page_config(page_title="Fraud Detection App", page_icon="💳", layout="centered")

# ------------------------------------------------------------------
# Load model (file .pkl harus ada di folder yang sama dengan app.py)
# ------------------------------------------------------------------
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

st.title("💳 Deteksi Fraud Transaksi Kartu Kredit")
st.write("Masukkan detail transaksi di bawah ini untuk memprediksi apakah transaksi tersebut **normal** atau **fraud**.")

with st.form("fraud_form"):
    col1, col2 = st.columns(2)

    with col1:
        amt = st.number_input("Jumlah Transaksi (amt, USD)", min_value=0.0, value=50.0, step=1.0)
        category = st.selectbox("Kategori Merchant", CATEGORIES)
        gender = st.selectbox("Jenis Kelamin Nasabah", ["M", "F"])
        state = st.selectbox("State Nasabah", STATES)
        city_pop = st.number_input("Populasi Kota Nasabah", min_value=0, value=50000, step=100)

    with col2:
        trans_date = st.date_input("Tanggal Transaksi", value=datetime.now())
        trans_time = st.time_input("Jam Transaksi", value=datetime.now().time())
        age = st.number_input("Usia Nasabah", min_value=15, max_value=100, value=35)
        distance_km = st.number_input("Jarak Nasabah ke Merchant (km)", min_value=0.0, value=50.0, step=1.0)

    submitted = st.form_submit_button("🔍 Prediksi")

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

    st.divider()
    if pred == 1:
        st.error(f"🚨 **FRAUD TERDETEKSI** (probabilitas fraud: {proba:.2%})")
    else:
        st.success(f"✅ **Transaksi Normal** (probabilitas fraud: {proba:.2%})")

    st.progress(min(float(proba), 1.0))

st.divider()
st.caption("Model machine learning ini dilatih menggunakan dataset Credit Card Transactions Fraud Detection (Kaggle) untuk keperluan tugas kelompok.")
