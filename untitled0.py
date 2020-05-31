import matplotlib.pyplot as plt
import numpy as np
import mglearn
from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris

iris = load_iris()

#print(iris.data)

X, y = make_blobs(n_samples=100, centers=4, n_features=2,random_state=0)
X1=X[:,0];
X2=X[:,1];

fig0, ax0 = plt.subplots()
plt.title('Исходные данные')
ax0.scatter(X1, X2)
plt.xlabel("1 признак")
plt.ylabel("2 признак")

fig1, ax1 = plt.subplots()
plt.title('Классы')
mglearn.discrete_scatter(X1, X2, y)
plt.legend(["Kaacc 0", "Кnacc 1", "Knacc 2", "Knacc 3"], loc="best")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
random_state=0

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

print("Правильностсь на обучающем наборе: %.2f" % clf.score(X_train, y_train))
print("Правильностсь на тестовом наборе: {:.2f}". format(clf.score(X_test, y_test)))

X_new = [2,4]
plt.plot(X_new[0],X_new[1],c="k" ,marker='*',markersize=10)
X_new =np.reshape(X_new, (1,2))

y_pred = clf.predict(X_new)[0]
print("Метка нового паттерна:", y_pred)
mx=1.01
my=1.01

plt.text(X_new[0,0]*mx, X_new[0,1]*my,
    "Пробный паттерн\n. Kaacc="+str(y_pred),
    color='k',
    fontsize=12)


 

 

