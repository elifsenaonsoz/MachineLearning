Naive Bayes ile İkili Sınıflandırma
Problem Tanımı
Bu projede, Gaussian Naive Bayes algoritması kullanılarak bir veri seti üzerinde ikili sınıflandırma gerçekleştirilmiştir. Model, hastaların sağlık verilerini analiz ederek, kalp hastalığı olup olmadıklarını tahmin etmektedir.

Proje kapsamında iki farklı yöntem karşılaştırılmıştır:

Scikit-learn GaussianNB modeli
Python ile sıfırdan yazılmış manuel Gaussian Naive Bayes modeli
Her iki modelin eğitim süresi, tahmin süresi ve doğruluk oranı analiz edilerek değerlendirilmiştir.

Veri Seti
Kaynak: archive.zip içerisindeki veri seti kullanılmıştır.
Özellikler: Hastaların yaş, kan basıncı, kolesterol seviyesi, EKG sonuçları, egzersiz sonrası ST depresyonu gibi sağlık verileri.
Hedef Değişken: Kalp hastalığı durumu
Eksik Veriler:
Sayısal değişkenler ortalama ile doldurulmuştur.
Kategorik değişkenler mod (en sık görülen değer) ile doldurulmuştur.
Özellik İşleme: One-hot encoding uygulanmıştır.
Veri, %70 eğitim - %30 test oranında ayrılmıştır.

Sonuçlar ve Karşılaştırma
İki modelin karşılaştırması, eğitim süresi, test süresi, doğruluk oranı ve karmaşıklık matrisi üzerinden yapılmıştır.

Scikit-learn GaussianNB Modeli
Doğruluk: %94.20
Eğitim Süresi: 0.001916 saniye
Tahmin Süresi: 0.000878 saniye
Karmaşıklık Matrisi:

lua
Kopyala
Düzenle
[[200  14]
 [  2  60]]
Yorum:

Modelin doğruluk oranı %94.20.
Eğitim süresi 0.0019 saniye.
Tahmin süresi 0.0008 saniye olup anlık tahminler için uygundur.
Yanlış sınıflandırmalar:
14 kişi kalp hastası olmadığı halde hastaymış gibi sınıflandırılmış.
2 kişi kalp hastası olduğu halde sağlıklı gibi tahmin edilmiş.
Custom GaussianNB (Manuel Versiyon)
Doğruluk: %94.93
Eğitim Süresi: 0.002992 saniye
Tahmin Süresi: 0.013963 saniye
Karmaşıklık Matrisi:

lua
Kopyala
Düzenle
[[200  14]
 [  0  62]]
Yorum:

Modelin doğruluğu %94.93 olup Scikit-learn modeline göre biraz daha iyi sonuç vermiştir.
Eğitim süresi 0.0029 saniye ile Scikit-learn modeline kıyasla biraz daha uzun sürmüştür.
Tahmin süresi 0.0139 saniye olup Scikit-learn modeline göre 15 kat daha yavaş çalışmaktadır.
Yanlış sınıflandırmalar:
14 kişi kalp hastası olmadığı halde hastaymış gibi tahmin edilmiş.
Hiçbir sağlıklı kişi yanlış sınıflandırılmamıştır, yani hastalara dair tahmin başarımı daha iyi olabilir.
Genel Karşılaştırma
Doğruluk açısından Custom GaussianNB modeli biraz daha iyi (%94.93 vs. %94.20).
Eğitim süresi açısından Scikit-learn modeli daha hızlıdır (0.0019s vs. 0.0029s).
Tahmin süresi açısından Scikit-learn modeli çok daha hızlıdır (0.0008s vs. 0.0139s).
Hata analizi açısından Custom modelin hastaları tanıma başarımı daha iyidir ancak test süresi oldukça uzundur.
Yanlış negatif hata oranı Custom modelde 0 olup, Scikit-learn modelinde 2 kişi yanlış negatif tahmin edilmiştir.
Sonuç olarak Scikit-learn modeli hız açısından çok daha avantajlı, ancak Custom model hasta sınıflandırmada daha iyi sonuç vermiştir.

Kaynakça
Veri Seti Kaynağı: UCI Makine Öğrenmesi Deposu - Heart Disease Dataset
Scikit-learn Naive Bayes Dokümantasyonu: Scikit-learn GaussianNB
