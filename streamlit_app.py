import streamlit as st
import pandas as pd
import pickle

# --- Model Yükleme ---
# Eğer 'models/xgboost_model.pkl' yolu doğruysa bu kısım çalışacaktır.
try:
    with open('models/xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Hata: 'models/xgboost_model.pkl' dosyası bulunamadı. Model dosyanızın yolunu kontrol edin.")
    model = None

st.title("🛒 Walmart Sales Forecasting App (YENİ)")

st.markdown("Modelinize uygun olan Çevresel ve Mağaza Tipi özelliklerini girerek haftalık satış tahmini yapın.")

if model is not None:
    # --- Kullanıcıdan Girdi Alma (Modelin Beklediği 7 Özellik) ---
    st.header("Çevresel ve Ekonomik Girdiler")
    
    # Ekonomik ve Hava Durumu Değişkenleri
    temperature = st.number_input("Temperature (°F)", min_value=-50.0, value=50.0)
    fuel_price = st.number_input("Fuel Price (USD)", min_value=1.0, value=3.0)
    cpi = st.number_input("CPI (Tüketici Fiyat Endeksi)", min_value=100.0, value=180.0)
    unemployment = st.number_input("Unemployment (İşsizlik Oranı)", min_value=0.0, max_value=20.0, value=8.0)

    st.header("Mağaza Tipi Kodlaması (One-Hot Encoded)")
    st.markdown("Lütfen mağazanın tipine (A, B veya C) göre sadece **bir** kutucuğu '1' olarak işaretleyin.")
    
    # Mağaza Tipi Kategorik Değişkenleri (One-Hot Encoded varsayılıyor)
    type_a = st.selectbox("Mağaza Tipi A (Type_A)", [0, 1])
    type_b = st.selectbox("Mağaza Tipi B (Type_B)", [0, 1])
    type_c = st.selectbox("Mağaza Tipi C (Type_C)", [0, 1])
    
    # --- Veri Çerçevesini Hazırlama ---
    # Sütun adları ve sırası modelin beklediği ile aynı olmalıdır!
    data = pd.DataFrame({
        'Temperature': [temperature],
        'Fuel_Price': [fuel_price],
        'CPI': [cpi],
        'Unemployment': [unemployment],
        'Type_A': [type_a],
        'Type_B': [type_b],
        'Type_C': [type_c]
    })

    # --- Tahmin Butonu ---
    if st.button("Predict Sales"):
        # XGBoost modelinin beklediği 7 özellik ile tahmin yap
        prediction = model.predict(data)[0]
        st.success(f"📌 Tahmini Haftalık Satış: **${prediction:,.2f}**")

# ---
