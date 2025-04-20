1. Matris Manipülasyonu, Özdeğer ve Özvektörlerin Makine Öğrenmesiyle İlişkisi
Matris Manipülasyonu Nedir?
Matris manipülasyonu, verilerin matris formatında işlenmesi ve dönüştürülmesidir. Örneğin bir veri kümesindeki değerleri normalize etmek, çarpmak, transpoze etmek gibi işlemler makine öğrenmesinde sıkça yapılır. Çünkü çoğu algoritma, verileri doğrudan matris yapıları üzerinde işler.

Özdeğer (Eigenvalue) ve Özvektör (Eigenvector) Nedir?
Özdeğer, bir matrisin bazı özel vektörler üzerindeki etkisini sadece ölçeklendirerek değiştirmesidir. Bu özel vektörlere de özvektör denir. Yani 
𝐴
𝑥
=
𝜆
𝑥
Ax=λx denkleminde, 
𝑥
x bir özvektör, 
𝜆
λ ise ona karşılık gelen özdeğerdir.

Makine Öğrenmesiyle İlişkisi
Makine öğrenmesinde en çok bilinen uygulamalardan biri PCA (Principal Component Analysis) algoritmasıdır. PCA, verideki varyansı en iyi açıklayan yönleri bulmak için kovaryans matrisinin özdeğer ve özvektörlerini hesaplar. Büyük özdeğere sahip olan bileşenler, verideki bilgiyi en çok taşıyan yönlerdir. Böylece yüksek boyutlu veri daha sade ama anlamlı hale getirilir.

Ayrıca bazı öğrenme algoritmalarında, modelin parametre matrisinin özelliklerini anlamak veya sistemin kararlılığını değerlendirmek için de özdeğerler analiz edilir.

Kaynaklar:

https://machinelearningmastery.com/introduction-matrices-machine-learning/

https://machinelearningmastery.com/introduction-to-eigendecomposition-eigenvalues-and-eigenvectors/
(Erişim: 7 Nisan 2025)

2. NumPy’ın linalg.eig() Fonksiyonu: Dokümantasyon ve Kaynak Kod İncelemesi
NumPy’ın linalg.eig() fonksiyonu, kare bir matrisin özdeğerlerini ve özvektörlerini hesaplamak için kullanılır. Özellikle sayısal çözümlerde hızlı ve güvenilir sonuçlar elde etmek amacıyla geliştirilmiştir.

Fonksiyonun temel kullanımı:

python
Kopyala
Düzenle
import numpy as np

A = np.array([[4, 2],
              [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)
eigenvalues: Özdeğerleri içeren NumPy dizisidir.

eigenvectors: Her sütunu bir özvektör olan matristir.

Dokümantasyon İncelemesi
Resmi dökümantasyona göre bu fonksiyon sadece kare matrisler için çalışır. Giriş olarak float veya kompleks sayı içeren matrisler kabul eder. Çıktıdaki özvektörler normalize edilmiştir, yani vektör normları 1 olacak şekilde ayarlanmıştır.

Özdeğerler karmaşık da olabilir, bu nedenle hem gerçek hem karmaşık matrisler desteklenir.

Kaynak:
https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html
(Erişim: 7 Nisan 2025)

Kaynak Kod İncelemesi
Fonksiyonun arka planında NumPy’ın doğrudan yazdığı Python kodu yoktur. Bunun yerine LAPACK adlı C tabanlı bir sayısal lineer cebir kütüphanesine bağlanır. NumPy içindeki linalg.py dosyası ve lapack_lite modülü üzerinden LAPACK’in geev fonksiyonu çağrılır.

Kaynak kod dizini:
https://github.com/numpy/numpy/tree/main/numpy/linalg
(Erişim: 7 Nisan 2025)

Bu yapı sayesinde:

Karakteristik polinom çözümü gibi yavaş yöntemler yerine daha stabil ve hızlı algoritmalar (örneğin QR ayrıştırması, Schur decomposition) kullanılır.

Özellikle büyük matrislerde daha verimli sonuçlar elde edilir.

Sonuç
NumPy eig() fonksiyonu, özdeğer ve özvektör hesaplamalarını pratik bir şekilde gerçekleştirmek için oldukça kullanışlıdır. Arka planda C dilinde çalışan LAPACK algoritmalarına dayanması nedeniyle hem performanslı hem de güvenilirdir.

3. Hazır Fonksiyonsuz Özdeğer Hesaplama ve Karşılaştırma
Bu bölümde, numpy.linalg.eig() fonksiyonunu kullanmadan bir matrisin özdeğerlerini ve özvektörlerini manuel olarak hesaplayan bir Python kodu çalıştırıldı. Ardından aynı matris için eig() fonksiyonu da kullanılarak sonuçlar karşılaştırıldı.

Kullanılan Matris:

python
Kopyala
Düzenle
A = np.array([[6, 2],
              [2, 3]])
Elle Hesaplama (Karakteristik Polinom Yöntemi)
Özdeğerler:

Determinant çözümü için:

det
(
𝐴
−
𝜆
𝐼
)
=
∣
6
−
𝜆
2
2
3
−
𝜆
∣
=
(
6
−
𝜆
)
(
3
−
𝜆
)
−
4
=
𝜆
2
−
9
𝜆
+
14
det(A−λI)= 
​
  
6−λ
2
​
  
2
3−λ
​
  
​
 =(6−λ)(3−λ)−4=λ 
2
 −9λ+14
Kökleri:

𝜆
1
=
7
,
𝜆
2
=
2
λ 
1
​
 =7,λ 
2
​
 =2
Özvektörler:

Her λ değeri için 
(
𝐴
−
𝜆
𝐼
)
𝑥
=
0
(A−λI)x=0 çözülerek bulunur. Örneğin, λ = 7 için:

𝐴
−
7
𝐼
=
[
−
1
2
2
−
4
]
⇒
𝑣
1
=
[
2
1
]
A−7I=[ 
−1
2
​
  
2
−4
​
 ]⇒v 
1
​
 =[ 
2
1
​
 ]
Normalize edilerek:
𝑣
^
1
=
1
5
[
2
1
]
v
^
  
1
​
 = 
5
​
 
1
​
 [ 
2
1
​
 ]

NumPy ile Hesaplama
python
Kopyala
Düzenle
eigenvalues, eigenvectors = np.linalg.eig(A)
Çıktılar:

lua
Kopyala
Düzenle
eigenvalues: [7. 2.]
eigenvectors:
[[0.8944 0.4472]
 [0.4472 -0.8944]]
Buradaki vektörler normalize edilmiştir ve elle bulunan vektörlerle aynı doğrultudadır.

Karşılaştırma ve Yorum
Elle yapılan hesaplamalar ile NumPy çıktıları birebir uyumludur. Tek fark, eig() fonksiyonu özvektörleri otomatik olarak normalize eder ve genelde yönlerini + veya – olarak farklı döndürebilir. Bu durum özvektörlerin anlamını değiştirmez çünkü vektör yönü sabit kaldığı sürece aynı doğrultuda kabul edilir.

Bu uygulama sayesinde, hem eig() fonksiyonunun verdiği sonuçların doğruluğu hem de temel özdeğer hesaplama yöntemleri anlaşılmış oldu.

Referans:
https://github.com/LucasBN/Eigenvalues-and-Eigenvectors
(Erişim: 7 Nisan 2025)
