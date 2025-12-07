
🛒 Walmart Satış Tahmini (Final Projesi)

Vehbi Ozan Güzel
İstanbul Atlas Üniversitesi — Yazılım Mühendisliği
MultiGroup Zero2End Machine Learning Bootcamp

🎯 Projenin Amacı

Bu projede amaç, Walmart mağazalarında haftalık satışları geçmiş veriler ve dış faktörler kullanarak tahmin edebilen bir makine öğrenimi modeli geliştirmektir.

Bu çalışma gerçek dünyada:

Stok optimizasyonu

Gelir tahmini

Kampanya planlama

Şube performans takibi

gibi kritik iş kararlarına destek sağlar.

📊 Kullanılan Veri Setleri

📌 Kaynak: Kaggle — Walmart Recruiting – Store Sales Forecasting
Veri seti 3 ana dosyadan oluşur:

Dosya	İçerik
train.csv	Mağaza, departman ve haftalık satış miktarları
features.csv	Hava durumu, ekonomik göstergeler, indirimler
stores.csv	Mağaza tipi ve büyüklük bilgileri

Toplam satır sayısı: 421.570+
Feature sayısı: 20+
Hedef değişken: Weekly_Sales

🛠️ Veri Ön İşleme Adımları

✔ Tarih formatı dönüştürüldü
✔ Mağaza ve çevresel bilgiler birleştirildi
✔ Eksik değerler analiz edildi → MarkDown kolonları 0 ile dolduruldu
✔ Feature Engineering yapıldı:

Yeni Özellik	Amaç
Yıl / Ay / Hafta	Mevsimsellik
Çeyrek	Finansal dönem etkisi
Gün Sırası	Tatil dönemlerine yakınlık
Mağaza Tipi (A/B/C → sayılaştırıldı)	Kategorik dönüşüm
🔍 Keşifsel Veri Analizi

Elde edilen iş içgörüleri:

Tatil haftalarında satış artıyor

A tipi mağazalar diğerlerinden daha yüksek satış yapıyor

Mağaza büyüklüğü satış ile pozitif ilişkili

Ekonomik değişkenlerin etkisi daha zayıf

📌 Tüm grafikler → docs/ klasöründe

🤖 Modelleme

Aşağıdaki modeller test edilmiştir:

Model	MAE	RMSE
RandomForest (Baseline)	1442	3680
Tuned RandomForest	1528	3888
XGBoost (Final Model)	1567	3179

🧠 Yorum:
➡ RMSE değerinde büyük gelişme sağladığı için
📌 XGBoost final model olarak seçildi

🎯 Değerlendirme ve İş Çıkarımları

Model özellikle mevsimsellik ve departman bazlı farkları iyi yakalamıştır

Bu yaklaşım stok hatalarını azaltabilir ve kârı artırabilir

Sonraki geliştirmelerde kampanya etkisi daha güçlü işlenebilir
"""
🧩 Proje Yapısı
walmart-sales-forecasting/
├── README.md
├── data/
├── notebooks/
├── src/
│   ├── pipeline.py
│   └── inference.py
├── models/
│   └── xgboost_model.pkl
└── docs/
    └── grafikler
    """

🚀 Nasıl Çalıştırılır?
1️⃣ Gerekli kütüphaneleri kurun:
pip install -r requirements.txt

2️⃣ Model tahmini almak için:
python src/inference.py
