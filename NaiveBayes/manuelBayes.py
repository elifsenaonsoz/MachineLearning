import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\Elif\Downloads\archive (25).zip', compression='zip')

for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

df = pd.get_dummies(df, drop_first=True)

if 'Outcome' in df.columns:
    target = 'Outcome'
else:
    target = df.columns[-1]

X = df.drop(columns=[target]).values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

class CustomGaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.array(X_c.mean(axis=0), dtype=np.float64)
            self.var[c] = np.array(X_c.var(axis=0), dtype=np.float64)
            self.priors[c] = X_c.shape[0] / X.shape[0]
            
    def predict(self, X):
        preds = []
        eps = 1e-9  
        for x in X:
            posteriors = {}
            for c in self.classes:
                mean_c = np.asarray(self.mean[c], dtype=np.float64)
             
                var_c = np.maximum(np.asarray(self.var[c], dtype=np.float64), eps)
                log_prior = np.log(self.priors[c])
                
                log_likelihood = np.sum(-0.5 * np.log(2.0 * np.pi * var_c)
                                          - ((x - mean_c) ** 2) / (2.0 * var_c))
                posteriors[c] = log_prior + log_likelihood
            preds.append(max(posteriors, key=posteriors.get))
        return np.array(preds)

model = CustomGaussianNB()
start_fit = time.time()
model.fit(X_train, y_train)
fit_time = time.time() - start_fit

start_pred = time.time()
y_pred_custom = model.predict(X_test)
pred_time = time.time() - start_pred

def compute_confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[int(true), int(pred)] += 1
    return cm

cm_custom = compute_confusion_matrix(y_test, y_pred_custom)
accuracy_custom = np.sum(y_test == y_pred_custom) / len(y_test)

print("Custom GaussianNB (Manuel Versiyon)")
print("Eğitim süresi: {:.6f} saniye".format(fit_time))
print("Test süresi: {:.6f} saniye".format(pred_time))
print("Accuracy: {:.4f}".format(accuracy_custom))
print("Karmaşıklık Matrisi:\n", cm_custom)

plt.figure(figsize=(6, 5))
im = plt.imshow(cm_custom, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Custom Gaussian Naive Bayes - Confusion Matrix")
plt.colorbar(im)
tick_marks = np.arange(len(np.unique(y_test)))
plt.xticks(tick_marks, np.unique(y_test))
plt.yticks(tick_marks, np.unique(y_test))
plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")

thresh = cm_custom.max() / 2.
for i in range(cm_custom.shape[0]):
    for j in range(cm_custom.shape[1]):
        plt.text(j, i, format(cm_custom[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm_custom[i, j] > thresh else "black")
plt.tight_layout()
plt.show()
