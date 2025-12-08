import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- Model Yükleme ---
try:
    with open('models/xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Hata: 'models/xgboost_model.pkl' dosyası bulunamadı. Model dosyanızın yolunu kontrol edin.")
    model = None

st.title("🛒 Walmart Sales Forecasting App (19 Özellikli)")

st.markdown("Tahmin için gerekli **tüm 19 özelliği** giriniz. Kategorik veriler (Type) sayısal koda dönüştürülecektir.")

# --- Label Encoding Sözlüğü ---
# Modelin eğitiminde kullanılan Label Encoding eşleşmesini doğru bildiğinizden emin olun!
# Varsayım: A=1, B=2, C=3
TYPE_MAPPING = {'A': 1, 'B': 2, 'C': 3}


if model is not None:
    
    # --- Girdiler (Önceki Koddan) ---
    st.header("Mağaza, Bölüm ve Tarih Bilgileri")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        store = st.number_input("Store", min_value=1, max_value=45, step=1, value=1)
        dept = st.number_input("Department", min_value=1, max_value=99, step=1, value=1)
        size = st.number_input("Store Size", min_value=0, value=150000)
    
    with col2:
        year = st.number_input("Year", min_value=2010, max_value=2013, value=2012)
        month = st.number_input("Month", min_value=1, max_value=12, value=6)
        week = st.number_input("Week", min_value=1, max_value=52, value=25)
    
    with col3:
        isholiday = st.selectbox("IsHoliday?", [0, 1])
        store_type_str = st.selectbox("Store Type (A, B, C)", ['A', 'B', 'C']) 
    
    st.markdown("---")
    
    # --- Ekonomik, Çevresel ve MarkDown Girdileri (Önceki Koddan) ---
    st.header("Ekonomik, Çevresel ve MarkDown Girdileri")
    
    col4, col5 = st.columns(2)
    with col4:
        temperature = st.number_input("Temperature (°F)", min_value=-50.0, value=50.0)
        fuel_price = st.number_input("Fuel Price (USD)", min_value=1.0, value=3.0)
        markdown1 = st.number_input("MarkDown1", min_value=0.0, value=0.0)
        markdown3 = st.number_input("MarkDown3", min_value=0.0, value=0.0)
        markdown5 = st.number_input("MarkDown5", min_value=0.0, value=0.0)
        
    with col5:
        cpi = st.number_input("CPI (Tüketici Fiyat Endeksi)", min_value=100.0, value=180.0)
        unemployment = st.number_input("Unemployment (İşsizlik Oranı)", min_value=0.0, max_value=20.0, value=8.0)
        markdown2 = st.number_input("MarkDown2", min_value=0.0, value=0.0)
        markdown4 = st.number_input("MarkDown4", min_value=0.0, value=0.0)

    # --- Otomatik Tarih Özelliklerini Hesaplama (Önceki Koddan) ---
    day_of_year = (week * 7) - 3
    day_of_year = max(1, day_of_year)
    
    quarter = 0
    if week <= 13:
        quarter = 1
    elif week <= 26:
        quarter = 2
    elif week <= 39:
        quarter = 3
    else:
        quarter = 4

    # --- KRİTİK ADIM: Kategorik Veriyi Sayısallaştırma ---
    # store_type_str ('A', 'B', 'C') değeri TYPE_MAPPING kullanılarak sayısal koda dönüştürülür.
    store_type_num = TYPE_MAPPING.get(store_type_str, 0) # Eşleşme bulunamazsa 0 kullanılır.


    # --- Veri Çerçevesini Hazırlama (Tüm 19 Özellik, Sayısal Tipte) ---
    data_dict = {
        # Modelin eğitim sırasında beklediği sıraya uyulmuştur.
        'Store': [store],
        'Dept': [dept],
        'IsHoliday': [isholiday],
        'Type': [store_type_num], # ARTIK SAYISAL
        'Size': [size],
        'Temperature': [temperature],
        'Fuel_Price': [fuel_price],
        'MarkDown1': [markdown1],
        'MarkDown2': [markdown2],
        'MarkDown3': [markdown3],
        'MarkDown4': [markdown4],
        'MarkDown5': [markdown5],
        'CPI': [cpi],
        'Unemployment': [unemployment],
        'Year': [year],
        'Month': [month],
        'Week': [week],
        'DayOfYear': [day_of_year],
        'Quarter': [quarter]
    }
    
    data = pd.DataFrame(data_dict)
    
    if st.button("Predict Sales"):
        try:
            # Tahmin yapılır
            prediction = model.predict(data)[0]
            st.success(f"📈 Tahmini Haftalık Satış: **${prediction:,.2f}**")
            
        except Exception as e:
            st.error("Tahmin yapılırken beklenmedik bir hata oluştu.")
            st.code(f"Hata Detayı: {e}")
