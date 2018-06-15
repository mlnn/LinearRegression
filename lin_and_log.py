import numpy as np
import random
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

n_samples = 100
np.random.seed(0)
X = np.random.normal(size=n_samples)
Y = np.random.normal(size=n_samples)

def threshold(z):
    if z > 0: return 1
    else: return 0

for i in range(100):
    Y[i] = threshold(X[i])

clf = linear_model.LogisticRegression()
X = X[:, np.newaxis]
clf.fit(X, Y)
lin = linear_model.LinearRegression()
lin.fit(X, Y)


plt.figure(1, figsize=(4, 3))
X_test = np.linspace(-5, 10, 300)
plt.ylabel('Y')
plt.xlabel('X')
plt.xticks(range(-5, 10))
plt.yticks([0, 0.5, 1])
plt.ylim(-.25, 1.25)
plt.xlim(-4, 10)

X = X.ravel()
plt.scatter(X, Y, color='black', zorder=1)

X_test = X_test[:, np.newaxis]

y_lin = lin.predict(X_test)
y_log_label = clf.predict(X_test)
y_log_probabilty = clf.predict_proba(X_test)[:,1]

#print(y_lin)
#print(y_log_label)
#print(y_log_probabilty)


plt.plot(X_test, y_lin, color='blue', linewidth=1)
plt.plot(X_test, y_log_label, color='red', linewidth=3)
plt.plot(X_test, y_log_probabilty, color='blue', linewidth=3)
plt.plot([-20,20],[0.5,0.5], color='green', linewidth=3)


plt.show()


Y_test = np.random.normal(size=300)

for i in range(300):
    Y_test[i] = threshold(X_test[i])


lin_score = lin.score(X_test, Y_test)
print(lin_score)
log_score = accuracy_score(y_log_label, Y_test)
print(log_score)



