import streamlit as st
import pandas as pd
import pickle
import numpy as np
import datetime

# --- Model Yükleme ---
try:
    with open('models/xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Hata: 'models/xgboost_model.pkl' dosyası bulunamadı. Model dosyanızın yolunu kontrol edin.")
    model = None

st.title("🛒 Walmart Sales Forecasting App (19 Özellikli)")

st.markdown("Tahmin için gerekli **tüm 19 özelliği** giriniz. Verileriniz modelin eğitim sırasına göre düzenlenecektir.")

if model is not None:
    
    # --- 1. Temel Girdiler ---
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
        # Modelin beklediği 'Type' (A, B, C) girdisi.
        store_type = st.selectbox("Store Type", ['A', 'B', 'C']) 
    
    st.markdown("---")
    
    # --- 2. Ekonomik ve Çevresel Girdiler ---
    st.header("Ekonomik ve Çevresel Veriler")
    
    col4, col5 = st.columns(2)
    with col4:
        temperature = st.number_input("Temperature (°F)", min_value=-50.0, value=50.0)
        fuel_price = st.number_input("Fuel Price (USD)", min_value=1.0, value=3.0)
    with col5:
        cpi = st.number_input("CPI (Tüketici Fiyat Endeksi)", min_value=100.0, value=180.0)
        unemployment = st.number_input("Unemployment (İşsizlik Oranı)", min_value=0.0, max_value=20.0, value=8.0)
        
    st.markdown("---")
    
    # --- 3. MarkDown Girdileri ---
    st.header("MarkDown İndirim Değerleri")
    st.caption("Genellikle sıfır veya pozitif değerlerdir. Geçerli değilse 0.0 giriniz.")
    
    col6, col7, col8, col9, col10 = st.columns(5)
    
    with col6:
        markdown1 = st.number_input("MarkDown1", min_value=0.0, value=0.0)
    with col7:
        markdown2 = st.number_input("MarkDown2", min_value=0.0, value=0.0)
    with col8:
        markdown3 = st.number_input("MarkDown3", min_value=0.0, value=0.0)
    with col9:
        markdown4 = st.number_input("MarkDown4", min_value=0.0, value=0.0)
    with col10:
        markdown5 = st.number_input("MarkDown5", min_value=0.0, value=0.0)
        
    # --- Otomatik Tarih Özelliklerini Hesaplama ---
    # DayOfYear ve Quarter, Week ve Year'dan türetilir.
    
    # Basit DayOfYear hesaplaması (haftanın ortasını varsayarak)
    day_of_year = (week * 7) - 3
    day_of_year = max(1, day_of_year) # 1'den küçük olmasın
    
    # Basit Quarter hesaplaması
    if week <= 13:
        quarter = 1
    elif week <= 26:
        quarter = 2
    elif week <= 39:
        quarter = 3
    else:
        quarter = 4

    # --- Veri Çerçevesini Hazırlama (Tüm 19 Özellik) ---
    # Sütun adları ve sırası modelin eğitim sırasıyla TAM OLARAK AYNI OLMALIDIR.
    data_dict = {
        'Store': [store],
        'Dept': [dept],
        'IsHoliday': [isholiday],
        'Type': [store_type], 
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
        'DayOfYear': [day_of_year], # Hesaplanan
        'Quarter': [quarter]        # Hesaplanan
    }
    
    data = pd.DataFrame(data_dict)
    
    if st.button("Predict Sales"):
        try:
            # Tahmin yapılır
            # NOT: Orijinal kodunuzda index [1] kullanılmıştı. Tek bir tahmin için [0] kullanılır. 
            # Eğer modeliniz bir array içinde tek bir değer döndürüyorsa [0] kullanın.
            prediction = model.predict(data)[0]
            st.success(f"📈 Tahmini Haftalık Satış: **${prediction:,.2f}**")
            
        except ValueError as e:
            st.error("Tahmin Hatası: Lütfen girdiğiniz tüm 19 özelliğin değerlerini ve modelinizin doğru yüklendiğini kontrol edin.")
            st.code(f"Hata Detayı: {e}")
