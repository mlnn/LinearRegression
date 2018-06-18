import pandas
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression

def sign(z):
    if (z[:,0] | z[:,1]) > 0: return 1
    else: return -1

data = pandas.read_csv('data-logistic.csv')

data_y = data.iloc[:,0]
y = data_y.as_matrix()

data_x = data.iloc[:, 1:3]
X = data_x.as_matrix()

clf = linear_model.LogisticRegression()
clf.fit(X, y)

np.random.seed(0)

#Xs = np.random.uniform(-10, 10, (1, 14))

#print(Xs[:,1])

X_test = np.random.uniform(-10, 10, (300, 2))


#print(X)
#print(X_test)

y_log = clf.predict(X_test)

#print(X_test[1,0])
for i in range(300):
    for y in range(2):
        Y_test[i] = sign(X_test[i])

#print(Y_test)
