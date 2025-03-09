Naive Bayes ile İkili Sınıflandırma

Problem Tanımı

Bu projede, Gaussian Naive Bayes algoritması kullanılarak bir veri seti üzerinde ikili sınıflandırma gerçekleştirilmiştir. Model, hastaların sağlık verilerini analiz ederek, kalp hastalığı olup olmadıklarını tahmin etmektedir.

Proje kapsamında iki farklı yöntem karşılaştırılmıştır:

Scikit-learn GaussianNB modeli

Python ile sıfırdan yazılmış manuel Gaussian Naive Bayes modeli

Her iki modelin eğitim süresi, tahmin süresi ve doğruluk oranı analiz edilmiştir.

Veri Seti
Kaynak: archive.zip içerisindeki veri seti kullanılmıştır.

Özellikler: Hastaların yaş, kan basıncı, kolesterol seviyesi, EKG sonuçları, egzersiz sonrası ST depresyonu gibi sağlık verileri.

Hedef Değişken: Kalp hastalığı durumu

Eksik Veriler:

Sayısal değişkenler ortalama ile doldurulmuştur.

Kategorik değişkenler mod (en sık görülen değer) ile doldurulmuştur.

Özellik İşleme: One-hot encoding uygulanmıştır.

Veri, %70 eğitim - %30 test oranında ayrılmıştır.

