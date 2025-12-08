# 🛒 Walmart Satış Tahmini (Final Projesi)

**Vehbi Ozan Güzel**  
**İstanbul Atlas Üniversitesi — Yazılım Mühendisliği**  
**MultiGroup Zero2End Machine Learning Bootcamp**

---

## 🎯 Projenin Amacı

Bu projede amaç, **Walmart** mağazalarında haftalık satışları geçmiş veriler ve dış faktörler kullanarak **tahmin edebilen** bir makine öğrenimi modeli geliştirmektir.

Bu çalışma gerçek dünyada:
- Stok optimizasyonu
- Gelir tahmini
- Kampanya planlama
- Şube performans takibi

gibi kritik iş kararlarına destek sağlar.

---

## 📊 Kullanılan Veri Setleri

Kaynak: Kaggle — *Walmart Recruiting – Store Sales Forecasting*  

Veri seti 3 ana dosyadan oluşur:

| Dosya | İçerik |
|------|--------|
| `train.csv` | Mağaza, departman ve haftalık satış miktarları |
| `features.csv` | Hava durumu, ekonomik göstergeler, indirimler |
| `stores.csv` | Mağaza tipi ve büyüklük bilgileri |

Toplam satır sayısı: **421.570+**  
Feature sayısı: **20+**  
Hedef değişken: **Weekly_Sales**

---

## 🛠️ Veri Ön İşleme Adımları

- Tarih formatı dönüştürüldü  
- Mağaza (`stores.csv`) ve çevresel (`features.csv`) bilgiler `train.csv` ile birleştirildi  
- Eksik değerler analiz edildi → MarkDown kolonları kampanya yok anlamına gelecek şekilde **0 ile dolduruldu**  
- Feature Engineering yapıldı:

| Yeni Özellik | Amaç |
|---|---|
| `Year`, `Month`, `Week` | Mevsimsellik ve dönemsel etkiler |
| `Quarter` | Finansal dönem etkisi |
| `DayOfYear` | Yıl içindeki konum, tatil dönemlerine yakınlık |
| `Type` (A/B/C → sayısal) | Kategorik değişkenin modele uygun hâle getirilmesi |

---

## 🔍 Keşifsel Veri Analizi

Elde edilen bazı iş içgörüleri:

- Tatil haftalarında satışlar **normal haftalara göre daha yüksek** çıkmıştır.  
- A tipi mağazalar, B ve C tipine göre **daha yüksek ortalama satış** yapmaktadır.  
- Mağaza büyüklüğü (`Size`) satış ile **pozitif ilişkili** görünmektedir.  
- Ekonomik değişkenlerin (CPI, Unemployment) etkisi diğer faktörlere göre daha sınırlıdır.

İlgili grafikler `docs/` klasöründe saklanmaktadır:
- `holiday_vs_sales.png`
- `store_type_vs_sales.png`
- `store_size_vs_sales.png`
- `feature_importance.png`

---

## 🤖 Modeller ve Sonuçlar

Aşağıdaki modeller denenmiştir:

| Model | MAE | RMSE |
|------|------:|------:|
| RandomForest (Baseline) | 1442 | 3680 |
| Tuned RandomForest | 1528 | 3888 |
| **XGBoost (Final Model)** | **1567** | **3179** |

**Yorum:**

- Tuned RandomForest, baseline modele göre daha iyi sonuç verememiştir (aşırı kısıtlama).  
- XGBoost modeli özellikle **RMSE değerini** önemli ölçüde düşürmüş ve en stabil sonuçları vermiştir.  
- Bu nedenle **final model olarak XGBoost seçilmiştir.**

---

## 🎯 Değerlendirme ve İş Çıkarımları

- Model, özellikle **mevsimsellik ve departman bazlı farklılıkları** iyi yakalamaktadır.  
- Tahminler, stok planlama ve kampanya yönetimi için yol gösterici olabilir.  
- Doğru kullanıldığında:
  - Rafta ürün kalmaması (stock-out) azaltılabilir
  - Gereksiz stok maliyetleri düşürülebilir
  - Gelir tahminleri iyileştirilebilir

---
Streamlit Deploy Linki:
https://walmart-sales-forecasting-bdpp4y7sckmpddaewrtmtx.streamlit.app/

---
## 🧩 Proje Yapısı

```text
walmart-sales-forecasting/
├── README.md
├── requirements.txt
├── data/
│   ├── train.csv
│   ├── stores.csv
│   └── features.csv
├── notebooks/
│   ├── 1_EDA.ipynb
│   ├── 2_Baseline_Model.ipynb
│   ├── 3_Feature_Engineering.ipynb
│   ├── 4_Model_Training.ipynb
│   ├── 5_Evaluation.ipynb
│   └── 6_Final_Pipeline.ipynb
├── src/
│   ├── pipeline.py        
│   └── inference.py       
├── models/
│   └── xgboost_model.pkl
└── docs/
    ├── holiday_vs_sales.png
    ├── store_type_vs_sales.png
    ├── store_size_vs_sales.png
    └── feature_importance.png
