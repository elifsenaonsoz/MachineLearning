# Logistic Regression ile İkili Sınıflandırma

## Problem Tanımı

Bu projede, Logistic Regression algoritması kullanılarak bir veri seti üzerinde **ikili sınıflandırma** gerçekleştirilmiştir. Model, banka müşterilerinin kişisel ve finansal bilgilerini analiz ederek, müşterilerin bir bankacılık kampanyasına **abone olup olmayacaklarını** tahmin etmektedir.

Proje kapsamında **iki farklı yöntem** karşılaştırılmıştır:

1. **Scikit-learn Logistic Regression modeli**
2. **Python ile sıfırdan yazılmış manuel Logistic Regression modeli (Gradient Descent)**

Her iki modelin **eğitim süresi, tahmin süresi, doğruluk oranı, precision, recall ve F1 skoru** analiz edilerek değerlendirilmiştir.

---

## Veri Seti

- **Kaynak:** Bank Marketing veri seti ([UCI Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)) kullanılmıştır.
- **Özellikler:** Müşterilerin yaşı, mesleği, medeni durumu, eğitim seviyesi, kredi durumu, kampanyayla temas sayısı gibi demografik ve kampanya bazlı veriler.
- **Hedef Değişken:** Bankacılık kampanyasına abone olma durumu (evet/hayır)
- **Eksik Veriler:**  
  - Sayısal değişkenler **ortalama** ile doldurulmuştur.  
  - Kategorik değişkenler **mod** (en sık görülen değer) ile doldurulmuştur.
- **Özellik İşleme:** Kategorik değişkenlere **Label Encoding** uygulanmıştır.
- **Veri Ölçeklendirme:** Özelliklere **StandardScaler** uygulanarak ölçeklendirme yapılmıştır.

Veri, **%80 eğitim - %20 test** oranında ayrılmıştır.

---

## Sonuçlar ve Karşılaştırma

İki modelin karşılaştırması, **eğitim süresi, test süresi, doğruluk oranı, precision, recall, F1 skoru ve karmaşıklık matrisi** üzerinden yapılmıştır.

### Scikit-learn Logistic Regression Modeli
- **Doğruluk:** %88.41
- **Precision:** %55.82
- **Recall:** %18.88
- **F1 Skoru:** %28.22
- **Eğitim Süresi:** Ölçeklendirme sonrası önemli ölçüde iyileşti.
- **Tahmin Süresi:** Ölçeklendirme sonrası önemli ölçüde iyileşti.

#### **Karmaşıklık Matrisi:**  
(Çıktılarda gösterilen `confusion_matrix_sklearn.png` görseline bakınız.)

### Custom Logistic Regression (Manuel Versiyon)
- **Doğruluk:** Model çalıştırılarak elde edilen değerlere göre belirtilmelidir.
- **Precision:** Model çalıştırılarak elde edilen değerlere göre belirtilmelidir.
- **Recall:** Model çalıştırılarak elde edilen değerlere göre belirtilmelidir.
- **F1 Skoru:** Model çalıştırılarak elde edilen değerlere göre belirtilmelidir.
- **Eğitim Süresi:** Model çalıştırılarak elde edilen değerlere göre belirtilmelidir.
- **Tahmin Süresi:** Model çalıştırılarak elde edilen değerlere göre belirtilmelidir.

#### **Karmaşıklık Matrisi:**  
(Çıktılarda gösterilen `confusion_matrix_manual.png` görseline bakınız.)

Scikit-learn modeli genel olarak hız ve kullanım kolaylığı açısından avantajlıyken, manuel olarak yazılan model, algoritmanın detaylarının anlaşılmasında ve kişiselleştirilebilirliğinde fayda sağlamaktadır.

---

## Kaynakça  

- **Veri Seti Kaynağı:** [UCI Makine Öğrenmesi Deposu - Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)  
- **Scikit-learn Logistic Regression Dokümantasyonu:** [Scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
