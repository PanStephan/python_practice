# Кластеризация методом К средних
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

# Генерация исхдных данных: используется функция make_bLobs
#X, y = make_blobs(n_samples=200, centers=5, n_features=2,random_state=0)
X1=X[:,0];
X2=X[:,1];

dataset = pd.read_csv('Skye.csv')
print("Dataset .head() \n ",dataset.head())
# входные значения
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values

# Визуализация данных
figi,_ = plt.subplots()
plt.plot(X1, X2,'ob')
plt.title('fig1')
plt.xlabel("Первый признак")
plt.ylabel("Второй признак")
# Создание модели вычисления К средних
from sklearn.cluster import KMeans
n_clusters=2
indom_state=0
mykmeans = KMeans(n_clusters=4, random_state=0)
# Обучение модели
mykmeans.fit(X)
c=mykmeans.cluster_centers_
y=mykmeans.labels_
print(y)
# Визуализация результатов кластеризации
mask = y
fig2,_ = plt.subplots()
plt.plot(X[mask, 0],X[mask, 1],'ob')
plt.plot(X[~mask, 0],X[~mask, 1],'sm')
plt.legend(['0','1'], loc='best')
plt.scatter(c[0, 0], c[0, 1], c='b' ,marker='*',s=200)
plt.scatter(c[1, 0], c[1, 1], c='m' ,marker='*',s=200)
# Тестирование - классификация нового паттерна
X_new = [2,3]
plt.plot (X_new[0],X_new[1],c='k' ,marker='*',markersize=10)
X_new =np.reshape(X_new, (1,2))
#X_new =[ [10, 5]]
#X_newe=np. array(X_new)
y_pred = mykmeans.predict(X_new) [0]
print('Метка паттерна',y_pred)
mx=1.01
my=1.01
plt.text(X_new[0,0]*mx, X_new[0,1]*my,
color='k',
fontsize=12)