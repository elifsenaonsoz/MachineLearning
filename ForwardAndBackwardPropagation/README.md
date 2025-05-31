#  Wine Quality Prediction - Neural Network from Scratch

Bu projede, kırmızı şarapların kimyasal özelliklerinden yola çıkarak kalite skorunu tahmin etmek amacıyla sıfırdan yazılmış bir **yapay sinir ağı (Neural Network)** modeli geliştirilmiştir. Proje, **YZM212 Makine Öğrenmesi** dersi kapsamında 6. laboratuvar ödevi olarak hazırlanmıştır.

---

##  Veri Seti

- **Kaynak:** [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
- **Tür:** Regresyon
- **Amaç:** `quality` (0–10 arası) değerini tahmin etmek
- **Giriş Özellikleri:** 11 adet kimyasal ölçüm (pH, alkol, sitrik asit vs.)

---

## Model Özellikleri

- **Yapı:** `input (11) → hidden (8, ReLU) → output (1, linear)`
- **Optimizasyon:** Gradient Descent
- **Ağırlık Başlatma:** He Initialization
- **Aktivasyon Fonksiyonu:** ReLU
- **Loss Fonksiyonu:** Mean Squared Error (MSE)
- **Epoch:** 300

---

##  Eğitim Performansı

| Metrik | Değer |
|--------|--------|
| MSE    | 0.45   |
| MAE    | 0.53   |
| R²     | 0.33   |

> Not: Bu metrikler normalize edilmiş çıktıların orijinal hâle çevrilmiş versiyonları üzerinde hesaplanmıştır.

---

##  İlk 10 Tahmin Sonucu

| Gerçek (quality) | Tahmin |
|------------------|--------|
| 5.00             | 5.36   |
| 6.00             | 5.72   |
| 5.00             | 5.61   |
| 6.00             | 5.71   |
| 5.00             | 5.41   |
| 5.00             | 5.73   |
| 5.00             | 5.71   |
| 6.00             | 5.82   |
| 6.00             | 5.85   |
| 6.00             | 5.60   |

> Bu tablo modelin çıktılarının oldukça dengeli ve anlamlı olduğunu göstermektedir.

---

##  Loss Grafiği

Eğitim süresince loss değerinin epoch'lara göre değişimi:

![Loss Grafiği](loss_plot.png)

---

##  Gereksinimler

```bash
pip install -r requirements.txt
