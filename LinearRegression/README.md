# Linear Regression - YZM212 4. Laboratuvar Çalışması

Bu çalışmada, iş deneyimi (yıl) ile maaş arasındaki doğrusal ilişkiyi modellemek amacıyla **Linear Regression** algoritması iki farklı şekilde uygulanmıştır:

1. **Least Squares Estimation (LSE)** yöntemiyle manuel çözüm
2. **Gradient Descent** algoritması ile parametre güncelleme

---

## Kullanılan Veri Seti

- **Kaynak:** [Kaggle - Simple Linear Regression Salary Data](https://www.kaggle.com/datasets?tags=13405-Linear+Regression&utm_source=chatgpt.com)
- **Özellikler:**
  - `YearsExperience`: Bağımsız değişken (iş deneyimi, yıl)
  - `Salary`: Bağımlı değişken (maaş)
- **Format:** CSV

---
##  Teorik Bilgi

### Linear Regression
Amaç, veri noktaları arasında en uygun doğrusal ilişkiyi (doğruyu) bulmaktır. Genel formülü:

    y_hat = θ₀ + θ₁ * x

### Least Squares Estimation (LSE)
Kapalı formül ile en uygun ağırlıklar şu şekilde hesaplanır:

    θ = (Xᵀ * X)⁻¹ * Xᵀ * y

### Gradient Descent
Amaç, hata fonksiyonunu minimize edecek ağırlıkları iteratif olarak bulmaktır.

    θ_j := θ_j - α * (1/m) * ∑(h_θ(xᶦ) - yᶦ) * x_jᶦ

---

##  Kullanılan Kütüphaneler

```bash
numpy
pandas
matplotlib
scikit-learn
