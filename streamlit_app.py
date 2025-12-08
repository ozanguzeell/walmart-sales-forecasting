import streamlit as st
import pandas as pd
import pickle

# --- Model Yükleme ---
try:
    with open('models/xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Hata: 'models/xgboost_model.pkl' dosyası bulunamadı. Model dosyanızın yolunu kontrol edin.")
    model = None

st.title("🛒 Walmart Sales Forecasting App (14 Özellikli)")

st.markdown("Tahmin için gerekli **tüm 14 özelliği** (Mağaza bilgileri, Tarih, Ekonomi ve Çevresel Veriler) giriniz.")

if model is not None:
    
    # --- 1. Sizin İlk Girdileriniz (7 Özellik) ---
    st.header("Mağaza, Bölüm ve Tarih Bilgileri")
    
    store = st.number_input("Store", min_value=1, max_value=45, step=1, value=1)
    dept = st.number_input("Department", min_value=1, max_value=99, step=1, value=1)
    size = st.number_input("Store Size", min_value=0, value=150000)
    year = st.number_input("Year", min_value=2010, max_value=2013, value=2012)
    month = st.number_input("Month", min_value=1, max_value=12, value=6)
    week = st.number_input("Week", min_value=1, max_value=52, value=25)
    isholiday = st.selectbox("IsHoliday?", [0, 1])

    st.markdown("---")
    
    # --- 2. Modelin Beklediği Eksik Girdiler (7 Özellik) ---
    st.header("Ekonomik, Çevresel ve Mağaza Tipi Girdileri")
    
    # Ekonomik ve Hava Durumu Değişkenleri
    temperature = st.number_input("Temperature (°F)", min_value=-50.0, value=50.0)
    fuel_price = st.number_input("Fuel Price (USD)", min_value=1.0, value=3.0)
    cpi = st.number_input("CPI (Tüketici Fiyat Endeksi)", min_value=100.0, value=180.0)
    unemployment = st.number_input("Unemployment (İşsizlik Oranı)", min_value=0.0, max_value=20.0, value=8.0)

    st.subheader("Mağaza Tipi Kodlaması (One-Hot Encoded)")
    st.markdown("Lütfen mağazanın tipine (A, B veya C) göre sadece **bir** kutucuğu '1' olarak işaretleyin.")
    
    # Mağaza Tipi Kategorik Değişkenleri
    type_a = st.selectbox("Mağaza Tipi A (Type_A)", [0, 1])
    type_b = st.selectbox("Mağaza Tipi B (Type_B)", [0, 1])
    type_c = st.selectbox("Mağaza Tipi C (Type_C)", [0, 1])
    
    # --- Veri Çerçevesini Hazırlama (Tüm 14 Özellik) ---
    # LÜTFEN AŞAĞIDAKİ SIRALAMANIN MODELİNİZİN EĞİTİM SIRASI İLE AYNI OLDUĞUNDAN EMİN OLUN.
    # Bu sıralama tahmini bir sıralamadır.
    data = pd.DataFrame({
        'Store': [store],
        'Dept': [dept],
        'Size': [size],
        'Year': [year],
        'Month': [month],
        'Week': [week],
        'IsHoliday': [isholiday],
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
        try:
            # Tahmin yapılır
            # NOT: Orijinal kodunuzda index [1] kullanılmış, tek bir tahmin için genellikle [0] kullanılır. 
            # Eğer sadece tek bir değer bekliyorsanız, [0] kullanın. Ben [0]'ı varsayıyorum.
            prediction = model.predict(data)[0]
            st.success(f"📈 Tahmini Haftalık Satış: **${prediction:,.2f}**")
            
        except ValueError as e:
            st.error(f"Tahmin Hatası: Özellik uyuşmazlığı devam ediyor. Lütfen DataFrame'deki **sütun adlarının** ve **sıralamasının** modelin eğitiminde kullanılan 14 özellik ile **tam olarak aynı** olduğunu kontrol edin.")
            st.code(f"Modelin Beklediği Özellikler: {model.get_booster().feature_names}")
            st.code(f"Sizin Sağladığınız Özellikler: {list(data.columns)}")
            st.code(f"Hata Detayı: {e}")
