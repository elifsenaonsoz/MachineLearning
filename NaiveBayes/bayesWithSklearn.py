import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay


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

clf = GaussianNB()
start_fit = time.time()
clf.fit(X_train, y_train)
fit_time = time.time() - start_fit

start_pred = time.time()
y_pred = clf.predict(X_test)
pred_time = time.time() - start_pred

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Scikit-learn GaussianNB")
print("Eğitim süresi: {:.6f} saniye".format(fit_time))
print("Test süresi: {:.6f} saniye".format(pred_time))
print("Accuracy: {:.4f}".format(accuracy))
print("Karmaşıklık Matrisi:\n", cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Scikit-learn Gaussian Naive Bayes - Confusion Matrix")
plt.show()
