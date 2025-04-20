1. Matris Manipülasyonu, Özdeğer ve Özvektörlerin Makine Öğrenmesiyle İlişkisi
Matris Manipülasyonu Nedir?
Matris manipülasyonu, verileri matrisler aracılığıyla temsil edip bu matrisler üzerinde çeşitli işlemler yapmaktır. Bu işlemler arasında transpoz alma, matris çarpımı, ters alma gibi işlemler yer alır. Makine öğrenmesinde veriler genellikle sayısal biçimde matrislere aktarılır ve bu matrisler üzerinden hesaplamalar yapılır. Özellikle model eğitimi, hata hesaplama ya da veri dönüştürme gibi işlemlerde matris manipülasyonu kaçınılmazdır.

Özdeğer (Eigenvalue) ve Özvektör (Eigenvector) Nedir?

Bir matrisin özdeğerleri, o matrisin bazı özel vektörler üzerindeki etkisini tanımlar. Bu özel vektörlere de özvektör denir. Matematiksel olarak, bir A matrisinin özdeğer ve özvektörleri 
𝐴
𝑥
=
𝜆
𝑥
Ax=λx eşitliğiyle ifade edilir. Burada 
𝑥
x özvektör, 
𝜆
λ ise ona karşılık gelen özdeğerdir.

Makine Öğrenmesiyle İlişkisi
Makine öğrenmesinde özellikle boyut indirgeme, veri ön işleme ve model optimizasyonu gibi konularda özdeğerler ve özvektörler oldukça önemli bir rol oynar. En bilinen uygulamalardan biri Principal Component Analysis (PCA)’dır. PCA, yüksek boyutlu verilerdeki en anlamlı yönleri bulmak için kovaryans matrisinin özdeğer ve özvektörlerini kullanır. Özdeğeri büyük olan bileşenler daha fazla bilgi içerdiği için veriyi bu doğrultuda dönüştürerek hem verimlilik sağlanır hem de gürültü azaltılmış olur.

Özdeğer ve özvektörler ayrıca makine öğrenmesi modellerinin kararlılığını ve davranışını analiz etmek için de kullanılır. Örneğin, bir ağırlık matrisinin özdeğerleri, modelin öğrenme sürecindeki yönelimleri ve sapmaları anlamamıza yardımcı olabilir.

2. NumPy’ın linalg.eig() Fonksiyonu: Dokümantasyon ve Kaynak Kod İncelemesi
NumPy’ın linalg.eig() fonksiyonu, kare (n x n boyutunda) bir matrisin özdeğerlerini ve özvektörlerini hesaplamak için kullanılır. Bu fonksiyon, lineer cebirin önemli konularından biri olan özdeğer ayrıştırmasını (eigendecomposition) sayısal olarak gerçekleştirir.

Fonksiyonun temel kullanımı aşağıdaki gibidir:

python
Kopyala
Düzenle
import numpy as np

A = np.array([[4, 2],
              [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)
eigenvalues, A matrisinin özdeğerlerini içeren bir NumPy dizisidir.

eigenvectors, her sütunu bir özvektör olan bir matristir. Her vektör, sırasıyla eigenvalues içindeki özdeğere karşılık gelir.

Dokümantasyon İncelemesi
Resmi NumPy dökümantasyonuna göre numpy.linalg.eig() fonksiyonu yalnızca kare matrisler üzerinde çalışır ve gerçel ya da karmaşık sayı içeren matrisleri destekler. Fonksiyonun döndürdüğü özvektörler normalize edilmiştir, yani her birinin normu 1 olacak şekilde ayarlanmıştır.

Fonksiyonun dökümantasyonda belirtilen temel özellikleri:

Kompleks özdeğer ve özvektör hesaplamalarını destekler.

Her özvektör, karşılık gelen özdeğerle aynı sıradadır.

Sayısal doğruluğu sağlamak için yüksek performanslı cebir kütüphaneleri kullanılır.

Kaynak:
https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html
(Erişim: 7 Nisan 2025)

Kaynak Kod İncelemesi
Fonksiyonun Python tarafında tanımlanmış hali aslında doğrudan çalışmaz. numpy.linalg.eig() fonksiyonu, alt düzeyde LAPACK kütüphanesinin geev fonksiyonuna yönlendirilir. Bu işlem NumPy’ın linalg modülü içindeki linalg.py dosyası ve lapack_lite modülü aracılığıyla gerçekleştirilir.

Kaynak kodlar şu dizin altında yer alır:
https://github.com/numpy/numpy/tree/main/numpy/linalg
(Erişim: 7 Nisan 2025)

Buradaki dosyalar incelendiğinde:

linalg.py içinde eig() fonksiyonunun çağrısının LAPACK bağlantılarına yönlendirildiği görülür.

Asıl özdeğer ve özvektör hesaplamaları, LAPACK’in C diliyle yazılmış geev algoritması kullanılarak gerçekleştirilir.

Bu yapı sayesinde büyük ve karmaşık matrislerde hızlı ve doğru hesaplamalar yapılabilir.

Fonksiyon, klasik yöntemlerde olduğu gibi karakteristik polinomun köklerini çözmek yerine, sayısal stabilitesi yüksek olan algoritmalarla (örneğin Schur ayrıştırması, QR yöntemi) işlemi gerçekleştirir.

Sonuç
NumPy’ın eig() fonksiyonu, Python’da özdeğer ve özvektör hesaplamak için en yaygın kullanılan araçlardan biridir. Dokümantasyonu açık ve detaylıdır. Arka planda ise, güçlü bir matematik kütüphanesi olan LAPACK ile bağlantılı olarak çalışır. Bu sayede yüksek boyutlu veriler üzerinde kararlı ve hızlı sonuçlar üretir. Makine öğrenmesi ve sayısal hesaplamalar içeren uygulamalarda bu fonksiyonun kullanımı oldukça yaygındır.

