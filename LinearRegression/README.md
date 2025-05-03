# 📈 Linear Regression - YZM212 4. Laboratuvar Çalışması

Bu çalışmada, iş deneyimi (yıl) ile maaş arasındaki doğrusal ilişkiyi modellemek amacıyla **Linear Regression** algoritması üç farklı şekilde uygulanmıştır:

1. **Least Squares Estimation (LSE)** yöntemiyle manuel çözüm
2. **Gradient Descent** algoritması ile parametre güncelleme
3. **Scikit-learn kütüphanesi** kullanılarak hazır modelle çözüm

---

## 📊 Kullanılan Veri Seti

- **Kaynak:** [Kaggle - Simple Linear Regression Salary Data](https://www.kaggle.com/datasets?tags=13405-Linear+Regression&utm_source=chatgpt.com)
- **Özellikler:**
  - `YearsExperience`: Bağımsız değişken (iş deneyimi, yıl)
  - `Salary`: Bağımlı değişken (maaş)
- **Format:** CSV

---

## 🧮 Teorik Bilgi

### 🔹 Linear Regression
Amaç, veri noktaları arasında en uygun doğrusal ilişkiyi (doğruyu) bulmaktır. Genel formülü:

\[
\hat{y} = \theta_0 + \theta_1 x
\]

### 🔹 Least Squares Estimation (LSE)
Kapalı formül ile en uygun ağırlıklar şu şekilde hesaplanır:

\[
\theta = (X^T X)^{-1} X^T y
\]

### 🔹 Gradient Descent
Amaç, hata fonksiyonunu minimize edecek ağırlıkları iteratif olarak bulmaktır.

\[
\theta_j := \theta_j - \alpha \cdot \frac{1}{m} \sum (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}
\]

---

## ⚙️ Kullanılan Kütüphaneler

```bash
numpy
pandas
matplotlib
scikit-learn
