# Naive Bayes ile İkili Sınıflandırma

## Problem Tanımı

Bu projede, Gaussian Naive Bayes algoritması kullanılarak bir veri seti üzerinde **ikili sınıflandırma** gerçekleştirilmiştir. Model, hastaların sağlık verilerini analiz ederek, **kalp hastalığı olup olmadıklarını** tahmin etmektedir.

Proje kapsamında **iki farklı yöntem** karşılaştırılmıştır:

1. **Scikit-learn GaussianNB modeli**
2. **Python ile sıfırdan yazılmış manuel Gaussian Naive Bayes modeli**

Her iki modelin **eğitim süresi, tahmin süresi ve doğruluk oranı** analiz edilerek değerlendirilmiştir.

---

## Veri Seti

- **Kaynak:** `archive.zip` içerisindeki veri seti kullanılmıştır.
- **Özellikler:** Hastaların yaş, kan basıncı, kolesterol seviyesi, EKG sonuçları, egzersiz sonrası ST depresyonu gibi sağlık verileri.
- **Hedef Değişken:** Kalp hastalığı durumu  
- **Eksik Veriler:**  
  - Sayısal değişkenler **ortalama** ile doldurulmuştur.  
  - Kategorik değişkenler **mod** (en sık görülen değer) ile doldurulmuştur.  
- **Özellik İşleme:** One-hot encoding uygulanmıştır.

Veri, **%70 eğitim - %30 test** oranında ayrılmıştır.

---

## Sonuçlar ve Karşılaştırma  

İki modelin karşılaştırması, **eğitim süresi, test süresi, doğruluk oranı ve karmaşıklık matrisi** üzerinden yapılmıştır.

### **1️⃣ Scikit-learn GaussianNB Modeli**
- **Doğruluk:** %94.20  
- **Eğitim Süresi:** 0.001916 saniye  
- **Tahmin Süresi:** 0.000878 saniye  

#### **Karmaşıklık Matrisi:**  
Actual \ Predicted	0	1
0	200	14
1	2	60
markdown
Kopyala
Düzenle

