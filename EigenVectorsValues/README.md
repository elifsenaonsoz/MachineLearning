# YZM212 - 3. Laboratuvar Değerlendirme Raporu

## 1. Matris Manipülasyonu, Özdeğer ve Özvektörlerin Makine Öğrenmesiyle İlişkisi

### Matris Manipülasyonu Nedir?

Matris manipülasyonu, verilerin matris formatında işlenmesi ve dönüştürülmesidir. Makine öğrenmesi algoritmaları genellikle verileri matrisler aracılığıyla işler. Örneğin veri çarpımı, normalizasyon, transpoz alma gibi işlemler bu başlık altında değerlendirilir.

### Özdeğer ve Özvektör Nedir?

Özdeğer (eigenvalue), bir matrisin bazı özel vektörler üzerindeki etkisini yalnızca ölçeklendirerek değiştirmesidir. Bu özel vektörlere özvektör (eigenvector) denir.

Yani bir `A` matrisi için `Ax = λx` denklemi sağlanıyorsa:

- `x`: özvektör  
- `λ`: özdeğer

### Makine Öğrenmesiyle İlişkisi

Makine öğrenmesinde özellikle **boyut indirgeme** alanında özdeğer ve özvektörler kullanılır. En yaygın örneklerden biri PCA (Principal Component Analysis) yöntemidir. PCA, verinin kovaryans matrisinin özdeğer ve özvektörlerini bularak, verideki en anlamlı yönleri çıkarır. Özdeğeri büyük olan yönler, verideki bilgiyi en iyi taşıyan bileşenlerdir.

Bunun dışında, bazı modellerde parametre matrisinin yapısını anlamak ya da sistemin kararlılığını analiz etmek için de bu kavramlardan yararlanılır.

### Kaynaklar

- <https://machinelearningmastery.com/introduction-matrices-machine-learning/>  
- <https://machinelearningmastery.com/introduction-to-eigendecomposition-eigenvalues-and-eigenvectors/>  
(Erişim: 7 Nisan 2025)

---

## 2. NumPy `linalg.eig()` Fonksiyonu: Dokümantasyon ve Kaynak Kod İncelemesi

NumPy’ın `linalg.eig()` fonksiyonu, kare (n x n) bir matrisin özdeğerlerini ve özvektörlerini hesaplar. Bu işlem özdeğer ayrıştırması (eigendecomposition) olarak bilinir.

### Fonksiyonun Temel Kullanımı

```python
import numpy as np

A = np.array([[4, 2],
              [1, 3]])
```

eigenvalues, eigenvectors = np.linalg.eig(A)
eigenvalues: Özdeğerleri içeren NumPy dizisidir.

eigenvectors: Her sütunu bir özvektör olan matristir.

Dokümantasyon Bilgisi
Fonksiyon sadece kare matrislerde çalışır.

Girdi olarak gerçel veya karmaşık sayılardan oluşan matrisler alır.

Özvektörler normalize şekilde (normu 1) döner.

Karmaşık özdeğerleri destekler.

Resmi dokümantasyon bağlantısı:
https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html
(Erişim: 7 Nisan 2025)

Kaynak Kod İncelemesi
Bu fonksiyon Python içinde doğrudan tanımlanmaz. Arka planda LAPACK isimli düşük seviyeli sayısal lineer cebir kütüphanesi ile çalışır. numpy/linalg/linalg.py ve lapack_lite modülleri aracılığıyla LAPACK’in geev fonksiyonu çağrılır.

Bu yapı sayesinde:

Özdeğerler klasik karakteristik polinom yöntemi yerine daha verimli QR ayrıştırması gibi algoritmalarla hesaplanır.

Yüksek boyutlu matrislerde sayısal doğruluk korunur.

Kaynak kod bağlantısı:
https://github.com/numpy/numpy/tree/main/numpy/linalg
(Erişim: 7 Nisan 2025)

Dokümantasyon Bilgisi
Fonksiyon sadece kare matrislerde çalışır.

Girdi olarak gerçel veya karmaşık sayılardan oluşan matrisler alır.

Özvektörler normalize şekilde (normu 1) döner.

Karmaşık özdeğerleri destekler.

Resmi dokümantasyon bağlantısı:
https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html
(Erişim: 7 Nisan 2025)

Kaynak Kod İncelemesi
Bu fonksiyon Python içinde doğrudan tanımlanmaz. Arka planda LAPACK isimli düşük seviyeli sayısal lineer cebir kütüphanesi ile çalışır. numpy/linalg/linalg.py ve lapack_lite modülleri aracılığıyla LAPACK’in geev fonksiyonu çağrılır.

Bu yapı sayesinde:

Özdeğerler klasik karakteristik polinom yöntemi yerine daha verimli QR ayrıştırması gibi algoritmalarla hesaplanır.

Yüksek boyutlu matrislerde sayısal doğruluk korunur.

Kaynak kod bağlantısı:
https://github.com/numpy/numpy/tree/main/numpy/linalg
(Erişim: 7 Nisan 2025)

3. Elle Özdeğer Hesaplama ve NumPy eig() ile Karşılaştırma
Aşağıda, eig() fonksiyonu kullanılmadan yapılan özdeğer hesaplaması ve numpy.linalg.eig() çıktısı karşılaştırılmıştır.

Kullanılan Matris
python
Kopyala
Düzenle
A = np.array([[6, 2],
              [2, 3]])
1) Karakteristik Polinom Yöntemi
text
Kopyala
Düzenle
det(A - λI) = (6 - λ)(3 - λ) - (2*2)
            = λ² - 9λ + 14
Kökler: λ₁ = 7, λ₂ = 2
2) Özvektör Hesabı
λ = 7 için (A - 7I) vektör uzayı:

text
Kopyala
Düzenle
[-1, 2]
[ 2, -4]
Bu doğrultuda bir özvektör: [2, 1]
Normalize edilirse: [2/√5, 1/√5] ≈ [0.8944, 0.4472]

3) NumPy Sonuçları
python
Kopyala
Düzenle
eigvals, eigvecs = np.linalg.eig(A)
Çıktı:

text
Kopyala
Düzenle
eigvals = [7. 2.]
eigvecs =
[[ 0.8944  0.4472]
 [ 0.4472 -0.8944]]
4) Karşılaştırma
Özdeğerler aynı: 7 ve 2.

Özvektörler normalize edilmiş halde dönüyor ve aynı doğrultuda.

Sadece işaret farkı olabilir (örneğin [x, y] ile [-x, -y] aynı yönü gösterir).

Kaynak
https://github.com/LucasBN/Eigenvalues-and-Eigenvectors
(Erişim: 7 Nisan 2025)

yaml
Kopyala
Düzenle

